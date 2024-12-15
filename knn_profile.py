import os.path
import sys
import time
import random
from typing import Callable
import faiss
import numpy as np
from scanpy import read_h5ad, AnnData
from constants import *
import logging
from functools import cache
import polars as pl
import matplotlib as mpl
from useful_functions import report_lost_entries
from pynndescent import NNDescent
from single_cell import SingleCell
import scann
from sklearn.neighbors import KDTree
from annoy import AnnoyIndex
import ngtpy
import nndescent
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from utils import run
from constants import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

infoHandler = logging.FileHandler('info.log', mode='w')
infoHandler.setLevel(logging.INFO)
logger.addHandler(infoHandler)

"""
By convention, refer to algorithms in databases as faiss_ivf, faiss_ivf_pq, pynndescent, annoy, nndescent, ngt_anng. 
"""


def load_green_sc(sc_file: str, retrieve: bool, save_to: str
                  ) -> (
        SingleCell):
    """
    Loads, QCs, and computes PCs for individual cells from single cell file
    specified.
    :param retrieve: load new and save if False, else load existing.
    :return SingleCell object with a key in obsm corresponding to a NP array
    with PCs.
    """
    if retrieve:
        return SingleCell(f'{save_to}.h5ad', num_threads=-1)
    else:
        sc = SingleCell(sc_file, num_threads=-1)
        print(sc.obs.columns)
        sc = sc.qc(
            custom_filter=pl.col('projid').is_not_null().and_(
                pl.col('cell.type.prob').ge(0.9).and_(
                    pl.col('is.doublet.df').not_())))
        sc.obs = sc.obs.with_columns(
            subset=pl.col('subset').cast(pl.String).replace({'CUX2+':
                                                                 'Excitatory'}))

        sc = sc.with_uns(QCed=True)
        sc = sc.hvg().normalize().PCA()

        # Filter out non-QCed PCs.
        # sc.obsm['PCs'] = sc.obsm['PCs'][sc.obs['passed_QC']]

        sc.save(f'{save_to}.h5ad', overwrite=True)

        return sc


def profile_pynn(cells: np.ndarray, index_neighbours: int, query: np.ndarray,
                 pruning_degree_multiplier: float,
                 num_neighbours: int, search_untrained: bool,
                 diversify_probability: float = 1) -> \
        tuple[
            np.ndarray, float, float]:
    """
    Monitors performance of NNDescent algorithm (Python implementation).
    :param index_neighbours: number of neighbours considered when constructing
    index (NOT how many are queried). More neighbours => more accuracy, less speed.
    :param pruning_degree_multiplier: pruning_degree_multiplier * num_neighbours
    is the maximum number of neighbours that may be kept at any point in index
    construction. Higher multiplier => more accuracy, but less speed.
    :param diversify_probability: probability that a redundant (long) edge is
    removed. Higher prob => less accuracy, but more speed.
    :param num_neighbours must be <= 30.
    :return: neighbours, indexing time, searching time.
    """
    start = time.time()
    index = NNDescent(
        cells,
        n_neighbors=index_neighbours,
        metric='euclidean',
        low_memory=True,
        pruning_degree_multiplier=pruning_degree_multiplier,
        diversify_prob=diversify_probability,
        compressed=True,
        verbose=True,
    )
    if not search_untrained:
        index_time = time.time() - start
        start = time.time()
        indices, distances = index.neighbor_graph
        search_time = time.time() - start
        return indices[:, 0:num_neighbours], index_time, search_time
    else:
        index.prepare()
        index_time = time.time() - start

        # Prepare must be called after because it forgets the neighbour graph.
        start = time.time()
        indices, distances = index.query(query, num_neighbours)
        search_time = time.time() - start
        return indices, index_time, search_time


def profile_faiss_ivf(cells: np.ndarray,
                      num_voronoi_cells: int, num_centroids: int,
                      num_neighbours: int,
                      query: np.ndarray, search_untrained: bool,
                      subsampling_factor: float | None = None) -> tuple[
    np.ndarray, float, float]:
    """
    Monitors performance of FAISS algorithm (IVF).
    :param cells: PCs corresponding to individual cells.
    :param num_voronoi_cells: number of voronoi cells that the space is split
    into. More voronoi cells => more accuracy, but less speed.
    :param num_centroids: number of centroids initialized in k-means clustering.
    More centroids => more accuracy, less speed.
    :param subsampling_factor: factor by which cells are subsampled.
    If none, set maximum cluster size to default.
    :return: neighbours, indexing time, searching time.
    """
    # Depth is dimensionality of space that PCs are in (number of PCs per
    # sample). Subtract one for ID column.
    num_cells = len(cells)
    depth = len(cells[0])

    start = time.time()
    index = faiss.IndexFlatL2(depth)  # the other index
    ivf_index = faiss.IndexIVFFlat(index, depth, num_centroids)
    ivf_index.train(cells)

    ivf_index.add(cells)
    index_time = time.time() - start

    ivf_index.nprobe = num_voronoi_cells

    # Set maximum cluster size to force subsampling.
    if subsampling_factor is not None:
        ivf_index.cp.max_points_per_centroid = int(
            np.ceil((num_cells * (1 - subsampling_factor)) / num_centroids
                    ))
        print(
            f'Number of cells is {num_cells}. Max points per cluster of {ivf_index.cp.max_points_per_centroid}, but actually {np.ceil(num_cells / num_centroids
                                                                                                                                      )}')

    # Modify
    start = time.time()
    distances, indices = ivf_index.search(query, num_neighbours)
    search_time = time.time() - start

    return indices, index_time, search_time


def profile_faiss_pq(cells: np.ndarray, vec_resolution: int,
                     num_voronoi_cells: int, num_centroids: int,
                     num_neighbours: int,
                     query: np.ndarray, search_untrained: bool) -> tuple[
    np.ndarray,
    float, float]:
    """
    Monitors performance of FAISS algorithm (IVFPQ).
    :param cells: PCs corresponding to individual cells.
    :param vec_resolution: quantized vector size such that vec_resolution | len(cells[0]).
    :param num_voronoi_cells: number of voronoi cells that the space is split
    into. More voronoi cells => more accuracy, but less speed.
    :param num_centroids: number of centroids initialized in k-means clustering.
    More centroids => more accuracy, less speed.
    :return: neighbours, indexing time, searching time.
    """
    # Filter out any PCs with null entries.
    # cells = cells[[not any(np.isnan(cell)) for cell in cells]]
    num_cells = len(cells)
    # Ideal number of centroids.
    # num_centroids = int(4 * np.sqrt(num_cells))

    # logger.debug(f'There are {num_cells_initial} cells, but '
    #              f'{num_cells_initial - num_cells} cells had null entries.')

    # Depth is dimensionality of space that PCs are in (number of PCs per
    # sample). Subtract one for ID column.
    depth = len(cells[0])

    # logger.debug(f'Dimensionality of each PC is {depth}.')
    # Number of bytes that each vector is compressed to after creation of
    # voronoi cells. Should divide depth!.
    # Number of bits that each vector is quantized to.
    resolved_vec_size = 8

    # Construct empty index for L2 norm and wrap with product quantizer.
    start = time.time()
    index = faiss.IndexFlatL2(depth)
    quantized_index = faiss.IndexIVFPQ(index, depth, num_centroids,
                                       vec_resolution, resolved_vec_size)
    # Train and add actual data.
    quantized_index.train(cells)
    quantized_index.add(cells)
    index_time = time.time() - start

    # Number of neighbours to visit on each search action.
    quantized_index.nprobe = num_voronoi_cells

    # Compute nearest neighbour and check which have itself.
    start = time.time()
    distances, indices = quantized_index.search(query, num_neighbours)
    search_time = time.time() - start

    return indices, index_time, search_time


def profile_faiss_sq(cells: np.ndarray, float_resolution: int,
                     num_voronoi_cells: int, num_centroids: int,
                     num_neighbours: int,
                     query: np.ndarray, search_untrained: bool) -> tuple[
    np.ndarray,
    float, float]:
    """
    Monitors performance of FAISS algorithm (IVFPQ).
    :param cells: PCs corresponding to individual cells.
    :param float_resolution: quantized float size (size to which each entry in a
    vector is compressed to) such that float_resolution | 32.
    :param num_voronoi_cells: number of voronoi cells that the space is split
    into. More voronoi cells => more accuracy, but less speed.
    :param num_centroids: number of centroids initialized in k-means clustering.
    More centroids => more accuracy, less speed.
    :return: neighbours, indexing time, searching time.
    """
    # Construct empty index for L2 norm and wrap with product quantizer.
    depth = len(cells[0])
    start = time.time()

    if float_resolution == 16:
        quantized_index = faiss.index_factory(depth,
                                              f'IVF{num_centroids},SQfp{float_resolution}')
    else:
        quantized_index = faiss.index_factory(depth,
                                              f'IVF{num_centroids},SQ{float_resolution}')

    # Train and add actual data.
    quantized_index.train(cells)
    quantized_index.add(cells)
    index_time = time.time() - start

    # Number of neighbours to visit on each search action.
    quantized_index.nprobe = num_voronoi_cells

    # Compute nearest neighbour.
    start = time.time()
    distances, indices = quantized_index.search(query, num_neighbours)
    search_time = time.time() - start

    return indices, index_time, search_time


def profile_faiss_imi(cells: np.ndarray,
                      num_voronoi_cells: int, num_centroids: int,
                      num_neighbours: int,
                      query: np.ndarray, search_untrained: bool):
    """
    Monitors performance of FAISS algorithm (IMI - multi index quantizer).
    :param cells: PCs corresponding to individual cells.
    :param num_voronoi_cells: number of voronoi cells that the space is split
    into. More voronoi cells => more accuracy, but less speed.
    :param num_centroids: number of centroids initialized in k-means clustering.
    More centroids => more accuracy, less speed.
    :return: neighbours, indexing time, searching time.
    """
    depth = len(cells[0])

    # Create index.
    start = time.time()
    index = faiss.index_factory(depth,
                                f'IMI2x{int(np.log2(num_centroids) // 2)},Flat')

    # Train and add actual data.
    index.train(cells)
    index.add(cells)
    index_time = time.time() - start

    # Number of neighbours to visit on each search action.
    index.nprobe = num_voronoi_cells

    # Compute nearest neighbour.
    start = time.time()
    distances, indices = index.search(query, num_neighbours)
    search_time = time.time() - start

    return indices, index_time, search_time


def profile_scann(cells: np.ndarray, num_leaves: int,
                  prop_leaves_searched: float,
                  index_neighbours: int,
                  query: np.ndarray, num_neighbours: int,
                  search_untrained: bool) -> tuple[
    np.ndarray, float, float]:
    """
    Monitors performance of SCANN algorithm.
    :param index_neighbours: number of neighbours considered when constructing index (NOT how many are queried). More neighbours => more accuracy, less speed.
    :param num_leaves: more leaves => more precision, but less speed.
    :param prop_leaves_searched: higher proportion of leaves searched => more precision but less speed. Must be less than or equal to num_leaves.
    :param num_neighbours: must be less than prop_leaves_searched * num_leaves
    """
    # anistropic_quant_threshold of 0.2 is used to tailor a loss function to the one in
    # paper that inspired scann, NaN uses standard reconstruction loss.
    num_leaves_to_search = int(prop_leaves_searched * num_leaves)
    # https://github.com/erikbern/ann-benchmarks/blob/3bb0474ebad6c64f4ef5317db3a3797eb1b58a36/ann_benchmarks/algorithms/scann/config.yml#L12 for config.
    start = time.time()
    builder = scann.scann_ops_pybind.builder(cells, index_neighbours,
                                             "squared_l2").tree(
        num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search,
        training_sample_size=len(cells)).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100)

    searcher = builder.build()
    index_time = time.time() - start

    start = time.time()
    indices, distances = searcher.search_batched(query)
    search_time = time.time() - start

    return indices[:, 0:num_neighbours], index_time, search_time


def profile_annoy(cells: np.ndarray, query: np.ndarray, num_trees: int,
                  num_neighbours: int, num_nodes_searched: int,
                  search_untrained: bool) -> tuple[
    np.ndarray, float, float]:
    """
    Monitors performance of Annoy algorithm.
    :param num_trees: more trees => more precision but less speed.
    :param num_neighbours:
    :param num_nodes_searched: more nodes searched => more precision but less
    speed. n_trees * n by default.
    :return: neighbours, index time and search time in a tuple.
    """
    depth = len(cells[0])

    start = time.time()
    index = AnnoyIndex(depth, 'euclidean')
    for i in range(len(cells)):
        index.add_item(i, cells[i])
    index.build(num_trees)
    index_time = time.time() - start

    # Get neighbours.
    start = time.time()
    neighbours = []
    for v in query:
        neighbours.append(index.get_nns_by_vector(v, num_neighbours,
                                                  search_k=num_nodes_searched))
    search_time = time.time() - start

    neighbours = np.array(neighbours)
    return neighbours, index_time, search_time


def profile_ngt_panng(cells: np.ndarray, num_neighbours: int, query: np.ndarray,
                      epsilon: float, search_untrained: bool,
                      edge_size_for_search: int, edge_size_for_creation: int,
                      num_forcedly_pruned_edges: int,
                      num_selectively_pruned_edges: int, search_range: float
                      ) -> \
        tuple[np.ndarray, float, float]:
    """
    Monitors performance of NGT-PANNG algorithm.
    :param epsilon: trade-off between accuracy and speed.
    :param edge_size_for_search: number of edges to search.
    :param edge_size_for_creation: number of neighbours initialised during index
    construction.
    :param num_forcedly_pruned_edges: maximum number of neighbours per vertex.
    Rest are pruned.
    :param num_selectively_pruned_edges: minimum number of neighbours per vertex.
    Note that, obviously, num_selectively_pruned_edges <= num_forcedly_pruned_edges.
    :param search_range: radius in which neighbours are searched for during
    index construction.
    Lower values for all of these parameters prioritize speed, but high values
    prioritize accuracy.
    :return:
    """
    depth = len(cells[0])

    with open(f'cells.txt', 'w') as f:
        for cell in cells:
            f.write(f'{' '.join(cell.astype(np.str_))}\n')

    # Initialise regular anng index.
    start = time.time()
    run(f'{NGT} create -i t '
        f'-g a '
        f'-S {edge_size_for_search} '
        f'-e {search_range} '
        f'-E {edge_size_for_creation} '
        f'-d {depth} '
        f'-o f '
        f'-D 1 '
        f'panng_index cells.txt')

    # Prune to create panng index.
    run(f'{NGT} prune '
        f'-e {num_forcedly_pruned_edges} '
        f'-s {num_selectively_pruned_edges} '
        f'panng_index')

    index = ngtpy.Index('panng_index')
    index_time = time.time() - start

    start = time.time()
    indices = []
    for v in query:
        # Return type is index, distance, which is effectively zip(indices, distances).
        # Since zip(a, b) returns a list of (a[i], b[i]), zip(*zip(a, b)) would
        # return [(a[0], a[1], ..., a[n - 1]), (b[0], b[1], ..., b[n - 1])]
        result = index.search(v, size=num_neighbours, epsilon=epsilon)
        curr_indices = list(list(zip(*result))[0])
        indices.append(curr_indices)
    search_time = time.time() - start

    index.close()
    run('rm -rf cells.txt panng_index')
    return np.array(indices), index_time, search_time


def profile_ngt_onng(cells: np.ndarray, num_neighbours: int, query: np.ndarray,
                     epsilon: float, search_untrained: bool,
                     edge_size_for_search: int, edge_size_for_creation: int,
                     in_degree: int,
                     out_degree: int, search_range: float
                     ) -> \
        tuple[np.ndarray, float, float]:
    """
    Monitors performance of NGT-ONNG algorithm.
    :param epsilon: trade-off between accuracy and speed.
    :param edge_size_for_search: number of edges to search.
    :param edge_size_for_creation: number of neighbours initialised during index
    construction.
    :param out_degree: lower bound on graph's out degree.
    :param in_degree: lower bound on graph's in degree.
    Note that, obviously, num_selectively_pruned_edges <= num_forcedly_pruned_edges.
    :param search_range: radius in which neighbours are searched for during
    index construction.
    Lower values for all of these parameters prioritize speed, but high values
    prioritize accuracy.
    :return:
    """
    depth = len(cells[0])

    with open(f'cells.txt', 'w') as f:
        for cell in cells:
            f.write(f'{' '.join(cell.astype(np.str_))}\n')

    # Initialise regular anng.
    start = time.time()
    run(f'{NGT} create -i t '
        f'-g a '
        f'-S {edge_size_for_search} '
        f'-e {search_range} '
        f'-E {edge_size_for_creation} '
        f'-d {depth} '
        f'-o f '
        f'-D 1 '
        f'anng-index cells.txt')

    # Reconstruct to onng by restricting in and out degree.
    run(f'{NGT} reconstruct-graph '
        f'-m S '
        f'-o {out_degree} '
        f'-i {in_degree} '
        f'anng-index onng-index')

    index = ngtpy.Index('onng-index')
    index_time = time.time() - start

    start = time.time()
    indices = []
    for v in query:
        # Return type is index, distance, which is effectively zip(indices, distances).
        # Since zip(a, b) returns a list of (a[i], b[i]), zip(*zip(a, b)) would
        # return [(a[0], a[1], ..., a[n - 1]), (b[0], b[1], ..., b[n - 1])]
        result = index.search(v, size=num_neighbours, epsilon=epsilon)
        curr_indices = list(list(zip(*result))[0])
        indices.append(curr_indices)
    search_time = time.time() - start

    index.close()
    run('rm -rf cells.txt anng-index onng-index')
    return np.array(indices), index_time, search_time


def profile_ngt_anng(cells: np.ndarray, num_neighbours: int, query: np.ndarray,
                     epsilon: float, search_untrained: bool,
                     edge_size_for_search: int, edge_size_for_creation: int,
                     graph_type: str) -> \
        tuple[np.ndarray, float, float]:
    """
    Monitors performance of NGT (ANNG) algorithm
    :param epsilon: trade-off between accuracy and speed.
    :param edge_size_for_search: number of edges to search.
    :param edge_size_for_creation: number of neighbours initialised during index
    construction.
    Lower values for all of these parameters prioritize speed, but high values
    prioritize accuracy.
    """
    depth = len(cells[0])
    start = time.time()
    ngtpy.create(path='anng_index', dimension=depth, distance_type="L2",
                 graph_type=graph_type,
                 edge_size_for_search=edge_size_for_search,
                 edge_size_for_creation=edge_size_for_creation)
    index = ngtpy.Index('anng_index')
    index.batch_insert(cells)
    index.save()
    index_time = time.time() - start

    indices = []
    start = time.time()
    for v in query:
        # Return type is index, distance, which is effectively zip(indices, distances).
        # Since zip(a, b) returns a list of (a[i], b[i]), zip(*zip(a, b)) would
        # return [(a[0], a[1], ..., a[n - 1]), (b[0], b[1], ..., b[n - 1])]
        result = index.search(v, size=num_neighbours, epsilon=epsilon)
        curr_indices = list(list(zip(*result))[0])
        indices.append(curr_indices)
        # logger.debug(indices)
    search_time = time.time() - start

    index.close()
    run('rm -rf anng_index')
    return np.array(indices), index_time, search_time


def profile_ngt_anng_default(cells: np.ndarray, num_neighbours: int,
                             query: np.ndarray,
                             epsilon: float, search_untrained: bool,
                             edge_size_for_search: int,
                             edge_size_for_creation: int) -> \
        tuple[np.ndarray, float, float]:
    return profile_ngt_anng(cells, num_neighbours, query, epsilon,
                            search_untrained, edge_size_for_search,
                            edge_size_for_creation, 'ANNG')


def profile_ngt_anng_ianng(cells: np.ndarray, num_neighbours: int,
                           query: np.ndarray,
                           epsilon: float, search_untrained: bool,
                           edge_size_for_search: int,
                           edge_size_for_creation: int) -> \
        tuple[np.ndarray, float, float]:
    return profile_ngt_anng(cells, num_neighbours, query, epsilon,
                            search_untrained, edge_size_for_search,
                            edge_size_for_creation, 'IANNG')


def profile_ngt_anng_ranng(cells: np.ndarray, num_neighbours: int,
                           query: np.ndarray,
                           epsilon: float, search_untrained: bool,
                           edge_size_for_search: int,
                           edge_size_for_creation: int) -> \
        tuple[np.ndarray, float, float]:
    return profile_ngt_anng(cells, num_neighbours, query, epsilon,
                            search_untrained, edge_size_for_search,
                            edge_size_for_creation, 'RANNG')


def profile_ngt_anng_rianng(cells: np.ndarray, num_neighbours: int,
                            query: np.ndarray,
                            epsilon: float, search_untrained: bool,
                            edge_size_for_search: int,
                            edge_size_for_creation: int) -> \
        tuple[np.ndarray, float, float]:
    return profile_ngt_anng(cells, num_neighbours, query, epsilon,
                            search_untrained, edge_size_for_search,
                            edge_size_for_creation, 'RIANNG')


def profile_nndescent(cells: np.ndarray, num_neighbours: int, query: np.ndarray,
                      num_trees: int, index_neighbours: int,
                      search_untrained: bool) -> tuple[
    np.ndarray, float, float]:
    """
    Monitors performance of NNDescent algorithm (C++ implementation).
    :param index_neighbours: number of neighbours considered when constructing
    index (NOT how many are queried). More neighbours => more accuracy, less speed.
    :param num_neighbours: the number of neighbours computed, must be <= 30.
    :param num_trees: the number of random projection trees to use when initially
    constructing the index. More trees => more speed (fewer updates required
    because each tree contains fewer points), but less precision.
    :return: neighbours, indexing time, searching time.
    """
    cells = cells.astype(np.float32)
    query = query.astype(np.float32)
    start = time.time()
    # logger.debug(cells.shape)
    # logger.debug(cells.dtype)
    index = nndescent.NNDescent(cells,
                                n_neighbors=index_neighbours,
                                n_trees=num_trees, n_threads=os.cpu_count())
    index_time = time.time() - start

    start = time.time()

    if not search_untrained:
        indices, distances = index.neighbor_graph
        search_time = time.time() - start
        return indices[:, 0:num_neighbours], index_time, search_time
    else:
        indices, distances = index.query(query, k=num_neighbours)
        search_time = time.time() - start
        return indices, index_time, search_time


def random_sample(cells: np.ndarray, pt_kept: float, retrieve: bool,
                  save_to: str | None) -> tuple[np.ndarray, list[int]]:
    """
    Randomly samples pt_kept% of cells. Saves if retrieve is false and save_to
    is not None, else loads from save_to. save_to cannot be None and retrieve
    cannot be True simultaneously.
    """
    if retrieve:
        return np.load(f'{save_to}.npy'), np.load(f'{save_to}_indices.npy')

    sample_indices = random.sample(list(range(len(cells))),
                                   int((pt_kept / 100) * len(cells)))
    sample_filter = [False] * len(cells)
    for index in sample_indices:
        sample_filter[index] = True

    sample = cells[sample_filter]

    if save_to is not None:
        np.save(f'{save_to}.npy', sample)
        np.save(f'{save_to}_indices.npy', sample_indices)
    return sample, sample_indices


def brute_force_knn(train_data: np.ndarray, test_data: np.ndarray,
                    num_neighbours: int,
                    retrieve: bool, algorithm: str, save_to: str | None,
                    leaf_size: int = None) -> np.ndarray:
    """
    :param save_to: saves to save_to if not None, else doesn't.
    :param retrieve: if retrieve is True, save_to can't be None.
    :param save_to: saves brute forced neighbours to the specified location iff save_to is not None
    """
    if retrieve:
        return np.load(save_to)

    if algorithm == 'kdtree':
        tree = KDTree(train_data, leaf_size=leaf_size)
        _, indices = tree.query(test_data, k=num_neighbours)
        if save_to is not None:
            np.save(save_to, indices)
        return indices
    elif algorithm == 'faiss':
        depth = len(train_data[0])
        index = faiss.index_factory(depth, 'Flat')
        index.add(train_data)
        _, indices = index.search(test_data, num_neighbours)
        if save_to is not None:
            np.save(save_to, indices)
        return indices


def trials(algorithm_name: str,
           algorithm: Callable[..., tuple[np.ndarray, float, float]],
           hyperparameters: dict[str, list], training_data: np.ndarray,
           num_neighbours: int, test_data: np.ndarray,
           compute_accuracy: bool, true_neighbours: np.ndarray | None,
           results_database: pl.DataFrame | None,
           search_untrained: bool) -> pl.DataFrame | None:
    """
    Varies every variable in hyperparameters while holding the other variables
    in hyperparameters fixed to benchmark.

    :param algorithm: function that implements a kNN algorithm.
    :param hyperparameters: contains all hyperparameters and corresponding
    values to be passed to a kNN algorithm, i.e., every parameter except 'cells'
    and 'num_neighbours'. Every key-value pair must satisfy that the value's
    middle element is the default value for the hyperparameter corresponding to
    the key, i.e., is held fixed when another hyperparameter is varied.
    :param compute_accuracy: if true, reports accuracy of nearest neighbours,
    else only speed.
    :param true_neighbours: true nearest neighbours, computed via BF. None if
    compute_accuracy is None, else not None.
    :param num_neighbours: number of nearest neighbours to query. Must match
    number of nearest neighbours in true_neighbours, i.e., must match
    true_neighbours.shape[1].
    :return: None
    """
    all_results_raw = {'Algorithm': [], 'Lenient Overlap': [],
                       'Strict Overlap': [], 'Self NN': [], 'Indexing Time': [],
                       'Search Time': [], 'Hyperparameters': [],
                       'Number of Neighbours': []}
    logger.info(algorithm_name)
    for var, var_values in hyperparameters.items():
        logger.info(f'Varying {var} with {var_values}...')
        for value in var_values:
            kwargs = {'cells': training_data, 'num_neighbours': num_neighbours,
                      'query': test_data,
                      'search_untrained': search_untrained}
            # Add fixed values.
            curr_hyperparameters = {fixed: fixed_vals[len(fixed_vals) // 2] for
                                    fixed, fixed_vals
                                    in hyperparameters.items() if fixed != var}
            # Add varying value.
            curr_hyperparameters.update({var: value})
            kwargs.update(
                curr_hyperparameters)
            logger.debug(kwargs)
            logger.info(f'Current value: {value}')
            neighbours, index_time, search_time = algorithm(**kwargs)
            logger.info(f'Search time: {search_time}')
            logger.info(f'Indexing time: {index_time}')
            all_results_raw['Indexing Time'].append(index_time)
            all_results_raw['Search Time'].append(search_time)
            all_results_raw['Hyperparameters'].append(str(curr_hyperparameters))
            all_results_raw['Algorithm'].append(algorithm_name)
            all_results_raw['Number of Neighbours'].append(num_neighbours)
            if compute_accuracy:
                # Returns the percentage of vertices for whom the algorithm
                # computed the vertex as its nearest neighbour.
                # logger.debug(neighbours[:, 0])
                mean_acc_nn = np.equal(neighbours[:, 0],
                                       range(len(test_data))).mean()
                # Compares neighbours position by position. Probably not a useful metric.
                mean_acc_bf_strict = np.equal(neighbours,
                                              true_neighbours).mean()
                overlap_counts = np.array([len(np.intersect1d(
                    cell_neighbour, cell_true_neighbour)) for
                    cell_neighbour, cell_true_neighbour
                    in
                    zip(neighbours, true_neighbours)])
                mean_acc_bf_lenient = sum(overlap_counts) / (
                        len(test_data) * num_neighbours)
                logger.info(
                    f'Percent of cells with themselves as NN: {mean_acc_nn * 100}')
                logger.info(
                    f'Positional overlap between NNs: {mean_acc_bf_strict * 100}')
                logger.info(f'Overlap between NNs: {mean_acc_bf_lenient * 100}')

                all_results_raw['Lenient Overlap'].append(mean_acc_bf_lenient)
                all_results_raw['Strict Overlap'].append(mean_acc_bf_strict)
                all_results_raw['Self NN'].append(mean_acc_nn)
            else:
                all_results_raw['Lenient Overlap'].append(None)
                all_results_raw['Strict Overlap'].append(None)
                all_results_raw['Self NN'].append(None)

        logger.info(' ')

    if results_database is not None:
        curr_df = pl.DataFrame(all_results_raw)
        # logger.debug(curr_df.schema)
        # logger.debug(results_database.schema)
        final_df = pl.concat(
            [results_database, curr_df],
            how='vertical_relaxed')
        return final_df


def compound_trials(algorithm_parameters: dict[str, tuple[
    Callable[..., tuple[np.ndarray, float, float]], dict[str, list]]],
                    training_data: np.ndarray,
                    true_neighbours: np.ndarray | None,
                    test_data: np.ndarray, num_neighbours: int,
                    results_database: pl.DataFrame | None,
                    search_untrained: bool,
                    compute_accuracy: bool = True) -> pl.DataFrame | None:
    for alg_name, func_params in algorithm_parameters.items():
        func, params = func_params
        results_database = trials(alg_name, func, params, training_data,
                                  num_neighbours,
                                  test_data, compute_accuracy, true_neighbours,
                                  results_database, search_untrained)

    return results_database


def plot_trials(df: pl.DataFrame, save_to: str) -> None:
    df = df.with_columns(
        total_time=pl.col('Indexing Time').add(pl.col('Search Time')))

    plot = so.Plot(df, x='Lenient Overlap',
                   y='total_time')
    plot = plot.add(so.Dot(alpha=0.75), color='Algorithm')

    # Adjust layout to make room for the legend
    f = plt.figure(figsize=(30, 10), dpi=100, layout="constrained")
    sf1, sf2 = f.subfigures(2, 1, hspace=1 / 2)

    # show() is required for compilation.
    plot.on(sf1).show()
    plot.facet(col='Algorithm').on(sf2).show()

    plt.savefig(save_to)


def load_pcs_green():
    # Compute PCs.
    if not os.path.exists(f'{KNN_DIR}/rosmap_sc_pcs.npy'):
        # Saves numpy file.
        if not os.path.exists(f'{KNN_DIR}/rosmap_sc.h5ad'):
            sc = load_green_sc(
                f'{PROJECT_DIR}/single-cell/Green/p400_qced_shareable.h5ad',
                False, f'{SCRATCH_DIR}/rosmap_sc')
        else:
            sc = load_green_sc(
                f'{SCRATCH_DIR}/single-cell/Green/p400_qced_shareable.h5ad',
                True, f'{KNN_DIR}/rosmap_sc')
        pcs = sc.obsm['PCs']
        np.save(f'{KNN_DIR}/rosmap_sc_pcs.npy', pcs)
    else:
        pcs = np.load(f'{KNN_DIR}/rosmap_sc_pcs.npy')

    return pcs

# TODO: quality control this.
def load_seaad_sc(sc_file: str, retrieve: bool, save_to: str
                  ) -> (
        SingleCell):
    """
    Loads, QCs, and computes PCs for individual cells from single cell file
    specified.
    :param retrieve: load new and save if False, else load existing.
    :return SingleCell object with a key in obsm corresponding to a NP array
    with PCs.
    """
    if retrieve:
        return SingleCell(f'{save_to}.h5ad', num_threads=-1)
    else:
        sc = SingleCell(sc_file, num_threads=-1)
        sc = sc.with_uns(QCed=True)
        print(sc.obs.columns)
        sc = sc.qc()
        sc = sc.hvg().normalize().PCA()

        # Filter out non-QCed PCs.
        # sc.obsm['PCs'] = sc.obsm['PCs'][sc.obs['passed_QC']]

        sc.save(f'{save_to}.h5ad', overwrite=True)

        return sc


def load_pcs(sc_data: str, dataset_name: str):
    # Compute PCs.
    if not os.path.exists(f'{KNN_DIR}/{sc_data}_pcs.npy'):
        # Saves numpy file.
        if not os.path.exists(f'{KNN_DIR}/{dataset_name}.h5ad'):
            if dataset_name == 'seaad':
                sc = load_green_sc(
                sc_data,
                False, f'{SCRATCH_DIR}/{dataset_name}')
            elif dataset_name == 'rosmap':
                sc = load_seaad_sc(
                    sc_data,
                    False, f'{SCRATCH_DIR}/{dataset_name}')
        else:
            if dataset_name == 'seaad':
                sc = load_green_sc(
                    sc_data,
                    True, f'{SCRATCH_DIR}/{dataset_name}')
            elif dataset_name == 'rosmap':
                sc = load_seaad_sc(
                    sc_data,
                    True, f'{SCRATCH_DIR}/{dataset_name}')

        pcs = sc.obsm['PCs']
        np.save(f'{KNN_DIR}/{dataset_name}_pcs.npy', pcs)
    else:
        pcs = np.load(f'{KNN_DIR}/{dataset_name}_pcs.npy')

    return pcs


def sample_data(pcs: np.ndarray, proportion: float) -> tuple[
    np.ndarray, list[int], np.ndarray, np.ndarray]:
    """
    Retrieves proportion percent sample from pcs.
    :param pcs: input array with rows representing data points and columns representing features / dimensions.
    :return: tuple of arrays corresponding to sample, further split into training and test sets.
    """
    if not os.path.exists(f'{KNN_DIR}/rosmap_pcs_random_sample_all.npy'):
        sample, sample_indices = random_sample(pcs, proportion, False,
                                               f'{KNN_DIR}/rosmap_pcs_random_sample_all')
    else:
        sample, sample_indices = random_sample(pcs, proportion, True,
                                               f'{KNN_DIR}/rosmap_pcs_random_sample_all')

    if not os.path.exists(f'{KNN_DIR}/rosmap_pcs_random_sample_train.npy'):
        sample_train_data, sample_test_data = split_data(pcs, 50,
                                                         f'{KNN_DIR}/rosmap_pcs_full_train.npy',
                                                         f'{KNN_DIR}/rosmap_pcs_full_test.npy')

        np.save(f'{KNN_DIR}/rosmap_pcs_random_sample_train.npy',
                sample_train_data)
        np.save(f'{KNN_DIR}/rosmap_pcs_random_sample_test.npy',
                sample_test_data)
    else:
        sample_train_data = np.load(
            f'{KNN_DIR}/rosmap_pcs_random_sample_train.npy')
        sample_test_data = np.load(
            f'{KNN_DIR}/rosmap_pcs_random_sample_test.npy')

    return sample, sample_indices, sample_train_data, sample_test_data


def split_data(pcs: np.ndarray, proportion: float, save_train_to: str | None,
               save_test_to: str | None) -> tuple[np.ndarray, np.ndarray]:
    """
    :param proportion: percentage to be used as training data.
    :param save_train_to: saves training data to the location specified iff not None.
    :param save_test_to: saves test data to the location specified iff not None.
    """
    if not os.path.exists(f'{KNN_DIR}/rosmap_pcs_train_full.npy'):
        train_data, train_data_indices = random_sample(pcs,
                                                       proportion,
                                                       False,
                                                       save_train_to)
        # Make remaining test data by just removing any vector in train_data.
        test_filter = [True] * len(pcs)
        for train_data_index in train_data_indices:
            test_filter[train_data_index] = False

        test_data = pcs[test_filter]

        if save_train_to is not None:
            np.save(save_train_to, train_data)

        if save_test_to is not None:
            np.save(save_test_to, test_data)

    else:
        train_data = np.load(f'{save_train_to}.npy')
        test_data = np.load(f'{save_test_to}.npy')

    return train_data, test_data


if __name__ == '__main__':
    pcs = load_pcs_green()

    # Define hyperparameters.
    faiss_pq_hyperparameters = {'vec_resolution': [5, 10, 25],
                                'num_voronoi_cells': [15, 30, 45],
                                'num_centroids': [750, 1500, 3000], }

    # Any resolution below 16 degrades accuracy significantly.
    faiss_sq_hyperparameters = {'float_resolution': [16],
                                'num_voronoi_cells': [15, 30, 45],
                                'num_centroids': [750, 1500, 3000], }

    # faiss_ivf_hyperparameters = {'num_voronoi_cells': [15, 30, 45],
    #                              'num_centroids': [750, 1500, 3000], }

    faiss_ivf_hyperparameters = {'num_voronoi_cells': [30],
                                 'num_centroids': [150, 300, 600, 4000, 5000], }

    faiss_imi_hyperparameters = {'num_voronoi_cells': [30],
                                 'num_centroids': [3500, 4000, 4500], }

    pynndescent_hyperparameters = {'index_neighbours': [5, 30, 50],
                                   'pruning_degree_multiplier': [0.5, 1.5, 3],
                                   'diversify_probability': [0.25, 0.5, 1]}

    nndescent_hyperparameters = {'index_neighbours': [10, 20, 25],
                                 'num_trees': [100]}

    scann_hyperparameters = {'num_leaves': [1000, 2000, 4000],
                             'prop_leaves_searched': [1 / 4, 1 / 2, 1],
                             'index_neighbours': [5, 10, 20]}

    ngt_anng_hyperparameters = {'epsilon': [0.1, 0.2, 0.3],
                                'edge_size_for_search': [20, 40, 80],
                                'edge_size_for_creation': [5, 10, 20]}

    # Remember that forcedly pruned edges is an upper bound and selectively pruned edges is a lower bound.
    ngt_panng_hyperparameters = {'epsilon': [0.1, 0.2, 0.3],
                                 'edge_size_for_search': [20, 40, 80],
                                 'edge_size_for_creation': [5, 10, 20],
                                 'num_forcedly_pruned_edges': [60, 90, 120],
                                 'num_selectively_pruned_edges': [40, 60, 80],
                                 'search_range': [0.05, 0.1, 0.2]}

    ngt_onng_hyperparameters = {'epsilon': [0.1, 0.2, 0.3],
                                'edge_size_for_search': [20, 40, 80],
                                'edge_size_for_creation': [5, 10, 20],
                                'in_degree': [60, 120, 180],
                                'out_degree': [5, 10, 20],
                                'search_range': [0.05, 0.1, 0.2]}

    annoy_hyperparameters = {'num_trees': [75, 150, 300],
                             'num_nodes_searched': [375, 750, 1500]}

    sample, sample_indices, sample_train_data, sample_test_data = sample_data(
        pcs, 10)

    # logger.debug(len(pcs))
    # logger.debug(len(pcs) // 2)

    ### ACCURACY TRIALS ###
    if False:
        # Compute true neighbours.
        if not os.path.exists(f'{KNN_DIR}/label_transfer_true_nn_sample.npy'):
            true_nearest_neighbours_untrained = brute_force_knn(
                sample_train_data,
                sample_test_data, 5,
                False,
                'faiss',
                f'{KNN_DIR}/label_transfer_true_nn_sample.npy',
                leaf_size=50)
        else:
            true_nearest_neighbours_untrained = brute_force_knn(
                sample_train_data,
                sample_test_data,
                10,
                True,
                'faiss',
                f'{KNN_DIR}/label_transfer_true_nn_sample.npy',
                leaf_size=50)

        if os.path.exists(f'{KNN_DIR}/umap_true_nn_sample.npy'):
            true_nearest_neighbours_trained = brute_force_knn(sample,
                                                              sample, 5,
                                                              True,
                                                              'faiss',
                                                              f'{KNN_DIR}/umap_true_nn_sample.npy',
                                                              leaf_size=50)
        else:
            true_nearest_neighbours_trained = brute_force_knn(sample,
                                                              sample, 5,
                                                              False,
                                                              'faiss',
                                                              f'{KNN_DIR}/umap_true_nn_sample.npy',
                                                              leaf_size=50)
        # algorithm_parameters = {
        #     'ngt_onng': (profile_ngt_onng, ngt_onng_hyperparameters),
        #     'ngt_panng': (profile_ngt_panng, ngt_panng_hyperparameters),
        #     'ngt_anng_default': (profile_ngt_anng_default,
        #                          ngt_anng_hyperparameters),
        #     'ngt_anng_ianng': (
        #         profile_ngt_anng_ianng, ngt_anng_hyperparameters),
        #     'ngt_anng_ranng': (
        #         profile_ngt_anng_ranng, ngt_anng_hyperparameters),
        #     'ngt_anng_rianng': (
        #         profile_ngt_anng_rianng, ngt_anng_hyperparameters),
        #     'nndescent': (profile_nndescent, nndescent_hyperparameters),
        #     'faiss_ivf': (profile_faiss_ivf, faiss_ivf_hyperparameters),
        #     'faiss_ivf_pq': (profile_faiss_pq,
        #                      faiss_pq_hyperparameters),
        #     'pynndescent': (profile_pynn,
        #                     pynndescent_hyperparameters),
        #     'annoy': (profile_annoy,
        #               annoy_hyperparameters), }
        algorithm_parameters = {
            'faiss_imi': (profile_faiss_imi, faiss_imi_hyperparameters)
        }

        ### TEST ON TRAINED DATA (UMAP APPLICATION).
        # Only for testing on training data results.
        if not os.path.exists(f'{KNN_DIR}/knn_profiles_test_on_trained'):
            test_on_trained_sample_results = pl.DataFrame(
                schema={'Algorithm': pl.String, 'Lenient Overlap': pl.Float32,
                        'Strict Overlap': pl.Float32, 'Self NN': pl.Float32,
                        'Indexing Time': pl.Float32,
                        'Search Time': pl.Float32, 'Hyperparameters': pl.String,
                        'Number of Neighbours': pl.Int64})
        else:
            test_on_trained_sample_results = pl.read_csv(
                f'{KNN_DIR}/knn_profiles_test_on_trained', separator='\t')

        # test_on_trained_sample_results = test_on_trained_sample_results.with_columns(
        #     Algorithm=pl.when(pl.col('Algorithm').eq('ngt_anng')).then(
        #         pl.lit('ngt_anng_default')))

        test_on_trained_sample_results = compound_trials(algorithm_parameters,
                                                         sample,
                                                         true_nearest_neighbours_trained,
                                                         sample,
                                                         5,
                                                         test_on_trained_sample_results,
                                                         False, True)

        test_on_trained_sample_results = test_on_trained_sample_results.sort(
            'Algorithm', maintain_order=True)

        test_on_trained_sample_results.write_csv(
            f'{KNN_DIR}/knn_profiles_test_on_trained', separator='\t')

        plot_trials(test_on_trained_sample_results,
                    f'{KNN_DIR}/knn_profiles_graphed_trained')

        ### TEST ON NEW DATA. ###
        if not os.path.exists(f'{KNN_DIR}/knn_profiles_test_on_untrained'):
            test_on_untrained_sample_results = pl.DataFrame(
                schema={'Algorithm': pl.String, 'Lenient Overlap': pl.Float32,
                        'Strict Overlap': pl.Float32, 'Self NN': pl.Float32,
                        'Indexing Time': pl.Float32,
                        'Search Time': pl.Float32, 'Hyperparameters': pl.String,
                        'Number of Neighbours': pl.Int64})
        else:
            test_on_untrained_sample_results = pl.read_csv(
                f'{KNN_DIR}/knn_profiles_test_on_untrained', separator='\t')

        # Querying non-trained data points.
        test_on_untrained_sample_results = compound_trials(algorithm_parameters,
                                                           sample_train_data,
                                                           true_nearest_neighbours_untrained,
                                                           sample_test_data,
                                                           5,
                                                           test_on_untrained_sample_results,
                                                           True, True)

        test_on_untrained_sample_results = test_on_untrained_sample_results.sort(
            'Algorithm', maintain_order=True)

        test_on_untrained_sample_results.write_csv(
            f'{KNN_DIR}/knn_profiles_test_on_untrained',
            separator='\t')

        plot_trials(test_on_untrained_sample_results,
                    f'{KNN_DIR}/knn_profiles_graphed_untrained')

    ### TIME TRIALS ###
    if True:
        # Sampling below 6000 centroids.
        faiss_ivf_hyperparameters = {'num_voronoi_cells': [30],
                                     'num_centroids': [1500], }

        faiss_sq_hyperparameters = {'num_voronoi_cells': [30],
                                    'float_resolution': [16],
                                    'num_centroids': [1500], }

        ngt_anng_hyperparameters = {'epsilon': [0.15],
                                    'edge_size_for_search': [40],
                                    'edge_size_for_creation': [10]}

        pynndescent_hyperparameters = {'index_neighbours': [30],
                                       'pruning_degree_multiplier': [1.5],
                                       'diversify_probability': [1]}

        nndescent_hyperparameters = {'index_neighbours': [30],
                                     'num_trees': [100]}

        algorithm_parameters = {
            'faiss_ivf': (profile_faiss_ivf, faiss_ivf_hyperparameters),
            'ngt_anng_default': (
                profile_ngt_anng_default, ngt_anng_hyperparameters),
            'nndescent': (profile_nndescent, nndescent_hyperparameters),
            'pynndescent': (profile_pynn, pynndescent_hyperparameters)
        }

        ### TEST ON TRAINED DATA (UMAP APPLICATION).
        # Get true nearest neighbours. Get first five for comparison.
        if not os.path.exists(f'{KNN_DIR}/umap_true_nn_full.npy'):
            true_nearest_neighbours_trained_full = brute_force_knn(pcs, pcs, 10,
                                                                   False,
                                                                   'faiss',
                                                                   f'{KNN_DIR}/umap_true_nn_full.npy')[
                                                   :, 0:5]
        else:
            true_nearest_neighbours_trained_full = brute_force_knn(pcs, pcs, 10,
                                                                   True,
                                                                   'faiss',
                                                                   f'{KNN_DIR}/umap_true_nn_full.npy')[
                                                   :, 0:5]

        if not os.path.exists(f'{KNN_DIR}/knn_profiles_test_on_trained_full'):
            test_on_trained_full_results = pl.DataFrame(
                schema={'Algorithm': pl.String, 'Lenient Overlap': pl.Float32,
                        'Strict Overlap': pl.Float32, 'Self NN': pl.Float32,
                        'Indexing Time': pl.Float32,
                        'Search Time': pl.Float32, 'Hyperparameters': pl.String,
                        'Number of Neighbours': pl.Int64})
        else:
            test_on_trained_full_results = pl.read_csv(
                f'{KNN_DIR}/knn_profiles_test_on_trained_full', separator='\t')

        test_on_trained_full_results = compound_trials(algorithm_parameters,
                                                       pcs,
                                                       true_nearest_neighbours_trained_full,
                                                       pcs,
                                                       5,
                                                       test_on_trained_full_results,
                                                       False, True)

        test_on_trained_full_results = test_on_trained_full_results.sort(
            'Algorithm', maintain_order=True)

        test_on_trained_full_results.write_csv(
            f'{KNN_DIR}/knn_profiles_test_on_trained_full', separator='\t')

        # plot_trials(test_on_trained_full_results, f'{KNN_DIR}/knn_profiles_test_on_trained_full')

        ### TEST ON NEW DATA (LABEL TRANSFER APPLICATION). ###
        # Split dataset 50-50 for test and training.
        full_train_data, full_test_data = split_data(pcs, 50,
                                                     f'{KNN_DIR}/rosmap_pcs_full_train',
                                                     f'{KNN_DIR}/rosmap_pcs_full_test')

        # Get true NNs. First 5 for comparison.
        if not os.path.exists(f'{KNN_DIR}/label_transfer_true_nn_full.npy'):
            true_nearest_neighbours_untrained_full = brute_force_knn(
                full_train_data, full_test_data, 10, False,
                'faiss',
                f'{KNN_DIR}/label_transfer_true_nn_full.npy')[:, 0:5]
        else:
            true_nearest_neighbours_untrained_full = brute_force_knn(
                full_train_data, full_test_data, 10, True,
                'faiss',
                f'{KNN_DIR}/label_transfer_true_nn_full.npy')[:, 0:5]

        if not os.path.exists(f'{KNN_DIR}/knn_profiles_test_on_untrained_full'):
            test_on_untrained_full_results = pl.DataFrame(
                schema={'Algorithm': pl.String, 'Lenient Overlap': pl.Float32,
                        'Strict Overlap': pl.Float32, 'Self NN': pl.Float32,
                        'Indexing Time': pl.Float32,
                        'Search Time': pl.Float32, 'Hyperparameters': pl.String,
                        'Number of Neighbours': pl.Int64})
        else:
            test_on_untrained_full_results = pl.read_csv(
                f'{KNN_DIR}/knn_profiles_test_on_untrained_full',
                separator='\t')

        # Querying non-trained data points.
        test_on_untrained_full_results = compound_trials(algorithm_parameters,
                                                         full_train_data,
                                                         true_nearest_neighbours_untrained_full,
                                                         full_test_data,
                                                         5,
                                                         test_on_untrained_full_results,
                                                         True,
                                                         True)

        test_on_untrained_full_results = test_on_untrained_full_results.sort(
            'Algorithm', maintain_order=True)

        test_on_untrained_full_results.write_csv(
            f'{KNN_DIR}/knn_profiles_test_on_untrained_full',
            separator='\t')

    test_on_trained_full_results = pl.read_csv(
        f'{KNN_DIR}/knn_profiles_test_on_trained_full', separator='\t')

    test_on_untrained_full_results = pl.read_csv(
        f'{KNN_DIR}/knn_profiles_test_on_untrained_full',
        separator='\t')

    all_full_results = pl.concat([test_on_trained_full_results.with_columns(
        Application=pl.lit('UMAP')),
        test_on_untrained_full_results.with_columns(
            Application=pl.lit(
                'Label Transfer'))]).with_columns(
        total_time=pl.col('Search Time') + pl.col('Indexing Time'))
    # plot = sns.boxplot(data=all_full_results, x='Application',
    #                    y='total_time',
    #                    hue='Algorithm')
    # fig = plot.get_figure()
    # fig.savefig(f'{KNN_DIR}/knn_profiles_graphed_times_full')

    # Plot UMAP.
    plot_trials(
        all_full_results.filter(
            pl.col('Lenient Overlap').is_not_null().and_(
                pl.col('Application').eq('UMAP'))),
        f'{KNN_DIR}/knn_profiles_graphed_umap_full')

    # Plot label transfer.
    plot_trials(
        all_full_results.filter(
            pl.col('Lenient Overlap').is_not_null().and_(
                pl.col('Application').eq('Label Transfer'))),
        f'{KNN_DIR}/knn_profiles_graphed_label_transfer_full')

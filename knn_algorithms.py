"""
Functions that call / implement algorithms for KNN passed into optimisation
framework or analysis framework.
"""
import time

import faiss
import numpy as np
import single_cell as sc


# TODO: update function to handle label transfer case.
def profile_faiss_michael(training_data: sc.SingleCell,
                          validation_data: np.ndarray, num_clusters: int,
                          min_clusters_searched: int,
                          max_clusters_searched: int,
                          num_candidates_per_neighbour: int,
                          num_kmeans_iterations, kmeans_barbar: int, num_neighbours: int):
    """
    Monitors performance of Michael's FAISS implementation.
    Note that this doesn't handle label transfer, i.e., querying neighbours of
    a subset of data in all of the data. This is why validation data is unused.
    :param training_data: single cell dataset.
    :param num_clusters: number of clusters that the search space is broken down
    into.
    :param min_clusters_searched: minimum number of clusters searched.
    :param max_clusters_searched: maximum number of clusters searched.
    :param num_candidates_per_neighbor: number of candidates searched per
    neighbor. True number may vary, less if too few cells in num_neighbours
    cells, more if need to complete searching cluster.
    :param num_kmeans_iterations: number of iterations run.
    :param kmeans_barbar: true if parallel initialisation vs random
    initialisation
    :return:
    """
    start = time.perf_counter()
    sc = training_data.neighbors(num_clusters=num_clusters,
                                 min_clusters_searched=min_clusters_searched,
                                 max_clusters_searched=max_clusters_searched,
                                 num_candidates_per_neighbor=num_candidates_per_neighbour,
                                 num_kmeans_iterations=num_kmeans_iterations,
                                 kmeans_barbar=bool(kmeans_barbar),
                                 num_neighbors=num_neighbours,
                                 overwrite=True)
    runtime = time.perf_counter() - start

    return sc.obsm['neighbors'], runtime


def profile_faiss_ivf(training_data: np.ndarray,
                      num_voronoi_cells: int, num_centroids: int,
                      num_neighbours: int,
                      validation_data: sc.SingleCell,
                      subsampling_factor: float | None = None) -> tuple[
    np.ndarray, float]:
    """
    Monitors performance of default FAISS algorithm (IVF).
    :param training_data: PCs corresponding to individual cells.
    :param num_voronoi_cells: number of voronoi cells that the space is split
    into. More voronoi cells => more accuracy, but less speed.
    :param num_centroids: number of centroids initialized in k-means clustering.
    More centroids => more accuracy, less speed.
    :param subsampling_factor: factor by which cells are subsampled.
    If none, set maximum cluster size to default.
    :return: neighbours, runtime.
    """
    # Depth is dimensionality of space that PCs are in (number of PCs per
    # sample). Subtract one for ID column.
    num_cells = len(training_data)
    depth = len(training_data[0])

    start = time.perf_counter()
    index = faiss.IndexFlatL2(depth)  # the other index
    ivf_index = faiss.IndexIVFFlat(index, depth, num_centroids)
    ivf_index.train(training_data)

    ivf_index.add(training_data)
    index_time = time.perf_counter() - start

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
    distances, indices = ivf_index.search(validation_data, num_neighbours)
    search_time = time.time() - start

    return indices, index_time + search_time

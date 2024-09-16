import os.path
import time
import random
from typing import Callable
import faiss
import numpy as np
import logging
from functools import cache
import polars as pl
from single_cell import SingleCell
from constants import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

infoHandler = logging.FileHandler('info.log', mode='w')
infoHandler.setLevel(logging.INFO)
logger.addHandler(infoHandler)


@cache
def load_sc(sc_file: str, retrieve: bool, save_to: str
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
        return SingleCell(f'{save_to}.h5ad')
    else:
        sc = SingleCell(sc_file)
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
        sc.obsm['PCs'] = sc.obsm['PCs'][sc.obs['passed_QC']]

        sc.save(f'{save_to}.h5ad', overwrite=True)

        return sc


def profile_faiss_ivf(cells: np.ndarray,
                      num_voronoi_cells: int, num_centroids: int,
                      num_neighbours: int,
                      query: np.ndarray, search_untrained: bool, ) -> tuple[
    np.ndarray, float, float]:
    """
    Monitors performance of FAISS algorithm (IVF).
    :param cells: PCs corresponding to individual cells.
    :param num_voronoi_cells: number of voronoi cells searched. More voronoi cells => more accuracy, but less speed.
    :param num_centroids: number of centroids initialized in k-means clustering.
    More centroids => more accuracy, less speed.
    """
    # Filter out any PCs with null entries.
    # cells = cells[[not any(np.isnan(cell)) for cell in cells]]
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

    start = time.time()
    index = faiss.IndexFlatL2(depth)  # the other index
    ivf_index = faiss.IndexIVFFlat(index, depth, num_centroids)
    ivf_index.train(cells)

    ivf_index.add(cells)
    index_time = time.time() - start

    ivf_index.nprobe = num_voronoi_cells
    start = time.time()
    distances, indices = ivf_index.search(query, num_neighbours)
    search_time = time.time() - start

    return indices, index_time, search_time


def random_sample(cells: np.ndarray, pt_kept: float, retrieve: bool,
                  save_to: str) -> tuple[np.ndarray, list[int]]:
    """
    Randomly samples pt_kept% of cells. Saves if retrieve is false, else loads
    from save_to.
    """
    if retrieve:
        return np.load(f'{save_to}.npy'), np.load(f'{save_to}_indices.npy')

    sample_indices = random.sample(list(range(len(cells))),
                                   int((pt_kept / 100) * len(cells)))
    sample_filter = [False] * len(cells)
    for index in sample_indices:
        sample_filter[index] = True

    sample = cells[sample_filter]

    np.save(f'{save_to}.npy', sample)
    np.save(f'{save_to}_indices.npy', sample_indices)
    return sample, sample_indices


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
    and 'num_neighbours'. Every key-value pair must satisfy that the value's middle element is the
    default value for the hyperparameter corresponding to the key, i.e., is held
    fixed when another hyperparameter is varied.
    :param compute_accuracy: if true, reports accuracy of nearest neighbours,
    else only speed.
    :param true_neighbours: true nearest neighbours, computed via BF. None if compute_accuracy is None, else not None.
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
        logger.debug(curr_df.schema)
        logger.debug(results_database.schema)
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


faiss_ivf_hyperparameters = {'num_voronoi_cells': [30, 45],
                             'num_centroids': [2500, 5000], }

algorithm_parameters = {
    'faiss_ivf': (profile_faiss_ivf, faiss_ivf_hyperparameters), }

if __name__ == '__main__':
    # Load PCs.
    if not os.path.exists(f'{KNN_DIR}/rosmap_sc_pcs.npy'):
        # Saves numpy file.
        if not os.path.exists(f'{KNN_DIR}/rosmap_sc.h5ad'):
            sc = load_sc(
                f'{PROJECT_DIR}/single-cell/Green/p400_qced_shareable.h5ad',
                False, f'{SCRATCH_DIR}/rosmap_sc')
        else:
            sc = load_sc(
                f'{SCRATCH_DIR}/single-cell/Green/p400_qced_shareable.h5ad',
                True, f'{KNN_DIR}/rosmap_sc')
        pcs = sc.obsm['PCs']
        np.save(f'{KNN_DIR}/rosmap_sc_pcs.npy', pcs)
    else:
        pcs = np.load(f'{KNN_DIR}/rosmap_sc_pcs.npy')

    # Get training data.
    if not os.path.exists(f'{KNN_DIR}/rosmap_pcs_full_train.npy'):

        full_train_data, full_train_data_indices = random_sample(pcs, 50,
                                                                 False,
                                                                 f'{KNN_DIR}/rosmap_pcs_full_train')
        # Make remaining test data by just removing any vector in train_data.
        test_filter = [True] * len(pcs)
        for train_data_index in full_train_data_indices:
            test_filter[train_data_index] = False

        full_test_data = pcs[test_filter]

        np.save(f'{KNN_DIR}/rosmap_pcs_full_test.npy', full_test_data)
        np.save(f'{KNN_DIR}/rosmap_pcs_full_train.npy', full_train_data)
    else:
        full_train_data = np.load(f'{KNN_DIR}/rosmap_pcs_full_train.npy')
        full_test_data = np.load(f'{KNN_DIR}/rosmap_pcs_full_test.npy')

    ### TEST ON NEW DATA. ###
    # Write results to DF for posterity's sake.
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
                                                     None,
                                                     full_test_data,
                                                     5,
                                                     test_on_untrained_full_results,
                                                     True, False)

    test_on_untrained_full_results = test_on_untrained_full_results.sort(
        'Algorithm', maintain_order=True)

    test_on_untrained_full_results.write_csv(
        f'{KNN_DIR}/knn_profiles_test_on_untrained_full',
        separator='\t')

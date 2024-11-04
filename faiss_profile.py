import os
from constants import *

from knn_profile import profile_faiss_ivf, load_pcs_green, sample_data, \
    compound_trials, split_data, random_sample, brute_force_knn
import numpy as np
from typing import Callable
from utils import run

import polars as pl
import optuna

faiss_ivf_hyperparameters = {'num_voronoi_cells': [30], }

algorithm_parameters = {
    'faiss_ivf': (profile_faiss_ivf, faiss_ivf_hyperparameters),
}


def get_num_centroids_michael(num_cells: int, centroid_scale_factor: float,
                              num_cells_per_centroid: int) -> int:
    return int(np.ceil(min(centroid_scale_factor * np.sqrt(num_cells),
                           num_cells / num_cells_per_centroid)))


def compute_accuracies(true_neighbours: np.ndarray,
                       testable_neighbours: np.ndarray, num_neighbours) -> \
        tuple[
            float, float, float]:
    # Is the first neighbour the element itself?
    mean_acc_nn = np.equal(testable_neighbours[:, 0],
                           range(len(testable_neighbours))).mean()

    # Compares neighbours position by position. Probably not a useful metric.
    mean_acc_bf_strict = np.equal(testable_neighbours,
                                  true_neighbours).mean()

    # Overall overlap regardless of ordering.
    overlap_counts = np.array([len(np.intersect1d(
        cell_neighbour, cell_true_neighbour)) for
        cell_neighbour, cell_true_neighbour
        in
        zip(testable_neighbours, true_neighbours)])

    mean_acc_bf_lenient = sum(overlap_counts) / (
            len(test_data) * num_neighbours)

    return mean_acc_nn, mean_acc_bf_strict, mean_acc_bf_lenient


### AUTOMATING HYPERPARAMETER OPTIMISATION ###
# Multi-objective function.
# TODO: how do I integrate data when objective only accepts one argument? Save and reload each time?
# TODO: choose ranges for hyperparameters.
def objective(trial: optuna.Trial) -> tuple[float, float]:
    train_data, validation_data, true_neighbours = np.load(
        TRAINING_DATASET), np.load(
        VALIDATION_DATASET), np.load(TRUE_NEIGHBOURS)

    # Determine hyperparameters for current trial.
    num_probes_range = (1, 60)
    num_centroids_range = (1, len(train_data))
    # Shouldn't more than double cluster size beyond default so lower bound is -1...
    subsampling_factor_range = (-1, 1)
    num_probes, num_centroids, subsampling_factor = (trial.suggest_int('num_probes', num_probes_range[0], num_probes_range[1]),
                                                     trial.suggest_int('num_centroids', num_centroids_range[0], num_centroids_range[1]),
                                                     trial.suggest_float('subsampling_factor', subsampling_factor_range[0], subsampling_factor_range[1]))
    # Note that search_untrained doesn't matter for FAISS since it always
    # uses the same method (no kNNG).
    neighbours, index_time, search_time = profile_faiss_ivf(train_data,
                                                            num_probes,
                                                            num_centroids, 5,
                                                            validation_data,
                                                            True)
    _, _, accuracy = compute_accuracies(true_neighbours, neighbours, 5)

    return index_time + search_time, accuracy


def auto_optimise_with_save(cells: np.ndarray, sample_pt: float, num_trials: int):
    sample, _ = random_sample(cells, sample_pt, False, None)
    train_data, validation_data = split_data(sample, 70, TRAINING_DATASET,
                                             VALIDATION_DATASET)
    _ = brute_force_knn(train_data, validation_data, 5, False, 'faiss', TRUE_NEIGHBOURS, None)
    study = optuna.create_study()
    study.optimize(objective, n_trials=num_trials)
    run(f'rm {TRAINING_DATASET} {VALIDATION_DATASET} {TRUE_NEIGHBOURS}')


# Try to prevent saving files and reloading. Would require redefiing objective
# to take in datasets as arguments.
# def auto_optimise(cells: np.ndarray, sample_pt: float, num_trials: int):
#     sample, _ = random_sample(cells, sample_pt, False, None)
#     train_data, validation_data = split_data(sample, 70, None,
#                                              None)
#     _ = brute_force_knn(train_data, validation_data, 5, False, 'faiss',
#                         None, None)
#     study = optuna.create_study()
#     study.optimize(objective, n_trials=num_trials)



if __name__ == '__main__':
    pcs = load_pcs_green()
    ### Auto-optimisation. ###
    # Try a couple of dataset sizes...
    sizes = [10, 25, 50, 100]
    for size in sizes:
        auto_optimise_with_save(pcs, size, 100)













    ### Manual optimisation. ###
    sample, sample_indices, sample_train_data, sample_test_data = sample_data(
        pcs, 50)

    # full_train_data, full_test_data = split_data(pcs, 50, f'{KNN_DIR}/rosmap_pcs_full_train', f'{KNN_DIR}/rosmap_pcs_full_test')

    train_data, test_data = sample_train_data, sample_test_data

    # Keep in mind that FAISS recommended scaling factors are 39 - 256 cells per cluster, 4 - 16 times sqrt(num cells).
    centroid_scaling_factor_list = list(range(4, 20, 4))
    num_cells_per_centroid_list = list(range(39, 256, 50))

    if not os.path.exists(f'{KNN_DIR}/faiss/results'):
        results = pl.DataFrame(schema={'centroid_scale_factor': pl.Float64,
                                       'num_cells_per_centroid': pl.Int64,
                                       'num_centroid_generator': pl.String,
                                       'time': pl.Float64,
                                       'application': pl.String})
    else:
        results = pl.read_csv(
            f'{KNN_DIR}/faiss/results', separator='\t')

    entries = {'centroid_scale_factor': [],
               'num_cells_per_centroid': [],
               'num_centroid_generator': [],
               'time': [],
               'application': []}

    for centroid_scale_factor in centroid_scaling_factor_list:
        for num_cells_per_centroid in num_cells_per_centroid_list:
            num_centroids = get_num_centroids_michael(len(train_data),
                                                      centroid_scale_factor,
                                                      num_cells_per_centroid)
            # Label transfer.
            _, index_time, search_time = profile_faiss_ivf(train_data,
                                                           30,
                                                           num_centroids, 5,
                                                           test_data,
                                                           True)
            entries['centroid_scale_factor'].append(centroid_scale_factor)
            entries['num_cells_per_centroid'].append(num_cells_per_centroid)
            entries['num_centroid_generator'].append(
                get_num_centroids_michael.__name__)
            entries['time'].append(search_time + index_time)
            entries['application'].append(LABEL_TRANSFER)

            # UMAP.
            _, index_time, search_time = profile_faiss_ivf(train_data,
                                                           30,
                                                           num_centroids, 5,
                                                           test_data,
                                                           True)
            entries['centroid_scale_factor'].append(
                centroid_scale_factor)
            entries['num_cells_per_centroid'].append(
                num_cells_per_centroid)
            entries['num_centroid_generator'].append(
                get_num_centroids_michael.__name__)
            entries['time'].append(search_time + index_time)
            entries['application'].append(UMAP)

    results = results.vstack(
        pl.DataFrame(entries, schema={'centroid_scale_factor': pl.Float64,
                                      'num_cells_per_centroid': pl.Int64,
                                      'num_centroid_generator': pl.String,
                                      'time': pl.Float64,
                                      'application': pl.String}))
    # Ensures that all rows are in a contiguous block of memory.
    results.rechunk()
    results.write_csv(file=f'{KNN_DIR}/faiss/results', separator='\t')

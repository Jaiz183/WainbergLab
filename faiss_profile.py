import os

import plotly.graph_objects

from constants import *

from knn_profile import profile_faiss_ivf, load_pcs_green, sample_data, \
    compound_trials, split_data, random_sample, brute_force_knn, load_pcs
import numpy as np
from typing import Callable
from utils import run

import polars as pl
import pandas as pd
import optuna

NUM_PROBES_RANGE = (1, 100)
SUBSAMPLING_FACTOR_RANGE = (0, 1)

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
            len(testable_neighbours) * num_neighbours)

    return mean_acc_nn, mean_acc_bf_strict, mean_acc_bf_lenient


### AUTOMATING HYPERPARAMETER OPTIMISATION ###
# def objective_with_save(trial: optuna.Trial):
#     train_data, validation_data, true_neighbours = np.load(
#         TRAINING_DATASET), np.load(
#         VALIDATION_DATASET), np.load(TRUE_NEIGHBOURS)
#
#     return objective(trial, train_data, validation_data, true_neighbours)


# Multi-objective function.
def objective(trial: optuna.Trial, training_dataset, validation_dataset,
              true_neighbours):
    # Determine hyperparameters for current trial.
    num_probes_range = (1, 100)
    # We'll use the default maximum number of clusters as the upper bound for now.
    num_centroids_range = (1, len(training_dataset) / 39)
    # Shouldn't more than double cluster size beyond default so lower bound is -1...
    subsampling_factor_range = (0, 1)
    num_probes, num_centroids, subsampling_factor = (
        trial.suggest_int('num_probes', num_probes_range[0],
                          num_probes_range[1], log=True),
        trial.suggest_int('num_centroids', num_centroids_range[0],
                          num_centroids_range[1], log=True),
        trial.suggest_float('subsampling_factor', subsampling_factor_range[0],
                            subsampling_factor_range[1]))
    # Note that search_untrained doesn't matter for FAISS since it always
    # uses the same method (no kNNG).
    neighbours, index_time, search_time = profile_faiss_ivf(training_dataset,
                                                            num_probes,
                                                            num_centroids, 5,
                                                            validation_dataset,
                                                            True,
                                                            subsampling_factor)
    _, _, accuracy = compute_accuracies(true_neighbours, neighbours, 5)

    return float(index_time + search_time), float(accuracy)


# def auto_optimise_with_save(cells: np.ndarray, sample_pt: float,
#                             num_trials: int) -> pl.DataFrame:
#     sample, _ = random_sample(cells, sample_pt, False, None)
#     train_data, validation_data = split_data(sample, 70, TRAINING_DATASET,
#                                              VALIDATION_DATASET)
#     _ = brute_force_knn(train_data, validation_data, 5, False, 'faiss',
#                         TRUE_NEIGHBOURS, None)
#     study = optuna.create_study(directions=['minimize', 'maximize'])
#     study.optimize(objective_with_save, n_trials=num_trials)
#     run(f'rm {TRAINING_DATASET} {VALIDATION_DATASET} {TRUE_NEIGHBOURS}')
#
#     return study.trials_dataframe()


# Try to prevent saving files and reloading. Would require redefining objective
# to take in datasets as arguments.
def auto_optimise(cells: np.ndarray, sample_pt: float,
                  num_trials: int) -> tuple[pl.DataFrame, optuna.Study]:
    sample, _ = random_sample(cells, sample_pt, False, None)
    train_data, validation_data = split_data(sample, 70, None,
                                             None)
    true_nn = brute_force_knn(train_data, validation_data, 5, False, 'faiss',
                              None, None)
    # Minimize runtime and maximize accuracy.
    study = optuna.create_study(directions=['minimize', 'maximize'])
    study.optimize(
        lambda trial: objective(trial, train_data, validation_data, true_nn),
        n_trials=num_trials)

    results: pd.DataFrame = study.trials_dataframe()
    columns_req = ['number', 'values_0', 'values_1', 'params_num_centroids',
                   'params_num_probes', 'params_subsampling_factor']
    results = results[columns_req]
    return pl.from_pandas(results).rename(
        {'values_0': 'runtime', 'values_1': 'accuracy'}), study


def save_plots(study, dataset_name: str, size: int):
    # Plot pareto.
    pareto: plotly.graph_objects.Figure = optuna.visualization.plot_pareto_front(
        study,
        target_names=["runtime",
                      "accuracy"])
    pareto.write_image(
        f'{KNN_DIR}/faiss/{dataset_name}_auto_tuning_{size}_sample_pareto.png')

    # Plot params corresponding to highest accuracy trials.
    parallel_accuracy: plotly.graph_objects.Figure = optuna.visualization.parallel_coordinates(
        target=lambda targets: targets.values[1])

    parallel_accuracy.write_image(
        f'{KNN_DIR}/faiss/{dataset_name}_auto_tuning_{size}_sample_parallel_accuracy.png')

    # Plot param importances.
    param_importances: plotly.graph_objects.Figure = optuna.visualization.plot_param_importances(
        study)

    param_importances.write_image(
        f'{KNN_DIR}/faiss/{dataset_name}_auto_tuning_{size}_param_importances.png')


if __name__ == '__main__':
    green_pcs = load_pcs(
        sc_data=f'{PROJECT_DIR}/single-cell/Green/p400_qced_shareable.h5ad',
        dataset_name='rosmap')

    seaad_pcs = load_pcs(
        sc_data=f'{PROJECT_DIR}/single-cell/SEAAD/SEAAD_DLPFC_RNAseq_all-nuclei.2024-02-13.h5ad',
        dataset_name='seaad')

    ### Auto-optimisation. ###
    # Try a couple of dataset sizes...
    sizes = [10, 25, 50, 100]
    for size in sizes:
        # First Green.
        results, green_study = auto_optimise(green_pcs, size, 100)
        results = results.sort('accuracy', descending=True)
        results.write_csv(
            f'{KNN_DIR}/faiss/green_auto_tuning_{size}_sample_results',
            separator='\t')
        save_plots(green_study, 'green', size)

        # Now SEAAD.
        results, seaad_study = auto_optimise(seaad_pcs, size, 100)
        results = results.sort('accuracy', descending=True)
        results.write_csv(
            f'{KNN_DIR}/faiss/seaad_auto_tuning_{size}_sample_results',
            separator='\t')
        save_plots(seaad_study, 'seaad', size)

    # ### Manual optimisation. ###
    # sample, sample_indices, sample_train_data, sample_test_data = sample_data(
    #     pcs, 50)
    #
    # # full_train_data, full_test_data = split_data(pcs, 50, f'{KNN_DIR}/rosmap_pcs_full_train', f'{KNN_DIR}/rosmap_pcs_full_test')
    #
    # train_data, test_data = sample_train_data, sample_test_data
    #
    # # Keep in mind that FAISS recommended scaling factors are 39 - 256 cells per cluster, 4 - 16 times sqrt(num cells).
    # centroid_scaling_factor_list = list(range(4, 20, 4))
    # num_cells_per_centroid_list = list(range(39, 256, 50))
    #
    # if not os.path.exists(f'{KNN_DIR}/faiss/results'):
    #     results = pl.DataFrame(schema={'centroid_scale_factor': pl.Float64,
    #                                    'num_cells_per_centroid': pl.Int64,
    #                                    'num_centroid_generator': pl.String,
    #                                    'time': pl.Float64,
    #                                    'application': pl.String})
    # else:
    #     results = pl.read_csv(
    #         f'{KNN_DIR}/faiss/results', separator='\t')
    #
    # entries = {'centroid_scale_factor': [],
    #            'num_cells_per_centroid': [],
    #            'num_centroid_generator': [],
    #            'time': [],
    #            'application': []}
    #
    # for centroid_scale_factor in centroid_scaling_factor_list:
    #     for num_cells_per_centroid in num_cells_per_centroid_list:
    #         num_centroids = get_num_centroids_michael(len(train_data),
    #                                                   centroid_scale_factor,
    #                                                   num_cells_per_centroid)
    #         # Label transfer.
    #         _, index_time, search_time = profile_faiss_ivf(train_data,
    #                                                        30,
    #                                                        num_centroids, 5,
    #                                                        test_data,
    #                                                        True)
    #         entries['centroid_scale_factor'].append(centroid_scale_factor)
    #         entries['num_cells_per_centroid'].append(num_cells_per_centroid)
    #         entries['num_centroid_generator'].append(
    #             get_num_centroids_michael.__name__)
    #         entries['time'].append(search_time + index_time)
    #         entries['application'].append(LABEL_TRANSFER)
    #
    #         # UMAP.
    #         _, index_time, search_time = profile_faiss_ivf(train_data,
    #                                                        30,
    #                                                        num_centroids, 5,
    #                                                        test_data,
    #                                                        True)
    #         entries['centroid_scale_factor'].append(
    #             centroid_scale_factor)
    #         entries['num_cells_per_centroid'].append(
    #             num_cells_per_centroid)
    #         entries['num_centroid_generator'].append(
    #             get_num_centroids_michael.__name__)
    #         entries['time'].append(search_time + index_time)
    #         entries['application'].append(UMAP)
    #
    # results = results.vstack(
    #     pl.DataFrame(entries, schema={'centroid_scale_factor': pl.Float64,
    #                                   'num_cells_per_centroid': pl.Int64,
    #                                   'num_centroid_generator': pl.String,
    #                                   'time': pl.Float64,
    #                                   'application': pl.String}))
    # # Ensures that all rows are in a contiguous block of memory.
    # results.rechunk()
    # results.write_csv(
    #     file=f'{KNN_DIR}/faiss/manual_tuning_results',
    #     separator='\t')

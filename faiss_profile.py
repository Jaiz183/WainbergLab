import os
import time
import typing

import plotly.graph_objects
from sklearn.neighbors import KDTree

from constants import *

from useful_functions import split_data
import numpy as np
import faiss

from single_cell import SingleCell

import polars as pl
import pandas as pd
import optuna

from knn_algorithms import profile_faiss_ivf, profile_faiss_michael

from networkx import DiGraph, all_topological_sorts
import timeout_decorator

import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

NUM_NEIGHBOURS = 5
APPLICATION = 'umap'
TRAINING_SPLIT = 70
NUM_TRIALS = 15
# Split is 50 seconds for kNN, 30 seconds for accuracy computation, 10 seconds
# for other stuff (topological ordering, flops, etc.)
TIMEOUT_INTERVAL = 120

# For pruning study so that unsuccessful trials aren't considered.
MINIMUM_ACCURACY = 0.9
MAXIMUM_RUNTIME = 50


def brute_force_knn(train_data: np.ndarray, test_data: np.ndarray,
                    num_neighbours: int,
                    retrieve: bool, algorithm: str, save_to: str | None,
                    leaf_size: int = None,
                    reject_self_neighbours: bool = False) -> np.ndarray:
    """
    Computes true NN using algorithm chosen.
    :param save_to: saves to save_to if not None, else doesn't.
    :param retrieve: if retrieve is True, save_to can't be None.
    :param algorithm: one of kdtree or faiss.
    :param reject_self_neighbours: avoid self neighbours by obtaining one extra
    neighbour and returning all but the closest neighbour.
    """
    if retrieve:
        return np.load(save_to)

    if algorithm == 'kdtree':
        if reject_self_neighbours:
            num_neighbours += 1

        tree = KDTree(train_data, leaf_size=leaf_size)
        _, indices = tree.query(test_data, k=num_neighbours)

        if reject_self_neighbours:
            indices = indices[:, 1:]

        if save_to is not None:
            np.save(save_to, indices)
        return indices

    elif algorithm == 'faiss':
        # Get an extra neighbour cuz of self neighbours.
        if reject_self_neighbours:
            num_neighbours += 1

        depth = len(train_data[0])
        index = faiss.index_factory(depth, 'Flat')
        index.add(train_data)
        _, indices = index.search(test_data, num_neighbours)

        if reject_self_neighbours:
            indices = indices[:, 1:]

        if save_to is not None:
            np.save(save_to, indices)
        return indices


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
        # print(sc.obs.columns)
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


# Old heuristic for number of centroids.
def get_num_centroids_michael(num_cells: int, centroid_scale_factor: float,
                              num_cells_per_centroid: int) -> int:
    return int(np.ceil(min(centroid_scale_factor * np.sqrt(num_cells),
                           num_cells / num_cells_per_centroid)))


def compute_accuracies(true_neighbours: np.ndarray,
                       testable_neighbours: np.ndarray,
                       num_neighbours) -> float:
    """
    Return accuracy of nearest neigbhours (position independent).
    """
    # Verify that indices are being passed in.

    # Overall overlap regardless of ordering.
    overlap_counts = np.array([len(np.intersect1d(
        cell_neighbour, cell_true_neighbour)) for
        cell_neighbour, cell_true_neighbour
        in
        zip(testable_neighbours, true_neighbours)])

    mean_acc_bf_lenient = sum(overlap_counts) / (
            len(testable_neighbours) * num_neighbours)

    return mean_acc_bf_lenient


# Multi-objective function.
def old_objective(trial: optuna.Trial, training_dataset, validation_dataset,
                  true_neighbours):
    # Determine hyperparameters for current trial.
    num_probes_range = (1, 100)
    # We'll use the default maximum number of clusters as the upper bound for now.
    num_centroids_range = (1, len(training_dataset) / 39)
    # Shouldn't increase cluster size beyond default...
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
    neighbours, runtime = profile_faiss_ivf(training_dataset,
                                            num_probes,
                                            num_centroids, 5,
                                            validation_dataset,
                                            subsampling_factor)
    accuracy = compute_accuracies(true_neighbours, neighbours, 5)

    return float(runtime), float(accuracy)


def compute_hyperparameter_ordering(hyperparams: dict) -> list[str]:
    """
    Computes ordering of hyperparameters according to dependencies between
    hyperparameters.
    :param hyperparams: hyperparameter ranges. Keys should correspond to
    names of hyperparams in algorithm. Values are a 4-tuple corresponding to
    min. val, max. val., parameter type (limited to int or float), and scale
    type (True if log, else false).
    Max. val. and min. val. can be strings corresponding to other
    hyperparameters or regular bounds. If they are, we use those hyperparameters
    as lower or upper bounds. Ensure that there are NO CYCLIC DEPENDENCIES.
    :return:
    """
    # Construct a graph of dependencies.
    # Compute nodes and edges.
    vertices = []
    edges = []
    for hyperparam in hyperparams:
        min_val, max_val, _, _ = hyperparams[hyperparam]
        # If any of the bounds are parameters (i.e., strings), add an edge.
        if type(min_val) is str:
            edges.append((min_val, hyperparam))
        if type(max_val) is str:
            edges.append((max_val, hyperparam))

        vertices.append(hyperparam)

    # Add separately because we might have vertices with no edges.
    dependency_graph = DiGraph()
    dependency_graph.add_nodes_from(vertices)
    dependency_graph.add_edges_from(edges)

    # Determine a topological sort of the orderings.
    return list(all_topological_sorts(dependency_graph))[0]


# Apply timeout decorator to time out kNN computation if it takes too long.
@timeout_decorator.timeout(TIMEOUT_INTERVAL,
                           timeout_exception=optuna.TrialPruned,
                           use_signals=True)
def objective_knn(trial: optuna.Trial, training_dataset,
                  validation_dataset,
                  hyperparam_ranges: dict, other_objective_params: dict,
                  algorithm: typing.Callable):
    """
    :param trial: optuna Trial object.
    :param training_dataset: dataset for constructing nearest neighbour graph.
    :param validation_dataset: dataset for computing accuracy.
    :param true_neighbours: actual neighbours to compute accuracy.
    :param hyperparam_ranges: hyperparameter ranges. Keys should correspond to
    names of hyperparams in algorithm. Values are a 4-tuple corresponding to
    min. val, max. val., parameter type (limited to int or float), and scale
    type (True if log, else false).
    Max. val. and min. val. can be strings corresponding to other
    hyperparameters. If they are, we use those hyperparameters are lower or
    upper bounds. Ensure that there are NO CYCLIC DEPENDENCIES.
    :param algorithm: nearest neighbour algorithm. Should take hyperparameters
    and non-hyperparameter inputs. Inputs are as follows:
    1) Hyperparameter inputs are named exactly the keys of hyperparam_ranges and
    their corresponding types
    2) validation_data
    3) training_data
    4) num_neighbours
    :param other_objective_params: contains true_neighbours key with true
    neighbours.
    :return: runtime and accuracy of algorithm w/ parameters.
    """
    true_neighbours = other_objective_params['true_neighbours']
    current_hyperparams = {}

    # Compute ordering such that we sample conditioned hyperparams before those
    # that depend on it for one of their bounds.
    hyperparam_ordering = compute_hyperparameter_ordering(hyperparam_ranges)

    # Determine hyperparameters for current trial.
    # print(hyperparam_ordering)
    for hyperparam in hyperparam_ordering:
        min_val, max_val, param_type, is_log = hyperparam_ranges[
            hyperparam]

        # If a bound depends on another parameter, obtain actual bound first.
        # Should raise an error if parameter hasn't been set already.
        if type(min_val) is str:
            min_val = trial.params[min_val]

        if type(max_val) is str:
            max_val = trial.params[max_val]

        # Suggest the right type.
        # If we're selecting max_clusters or min_clusters, make sure that their
        # values are below 2750 to prevent segfault.
        if hyperparam == 'max_clusters_searched' or \
                hyperparam == 'min_clusters_searched':
            current_hyperparams[hyperparam] = trial.suggest_int(
                hyperparam, min_val,
                min(max_val, 2749),
                log=is_log)
        elif param_type == 'int':
            current_hyperparams[hyperparam] = trial.suggest_int(
                hyperparam, min_val,
                max_val,
                log=is_log)
        else:
            current_hyperparams[hyperparam] = trial.suggest_float(
                hyperparam, min_val,
                max_val,
                log=is_log)

    print(f'Hyperparameters: {current_hyperparams}')
    # Add on non-hyperparameters
    current_hyperparams['training_data'], current_hyperparams[
        'validation_data'], current_hyperparams[
        'num_neighbours'] = training_dataset, validation_dataset, NUM_NEIGHBOURS

    # Compute objective values.
    neighbours, runtime = algorithm(
        **current_hyperparams)

    start = time.perf_counter()
    accuracy = compute_accuracies(true_neighbours, neighbours,
                                  NUM_NEIGHBOURS)
    print(
        f'Computing accuracy took {time.perf_counter() - start:.3f} seconds...')
    return float(runtime), float(accuracy)


# TODO: update function after updating Michael's implementation to handle label
#  transfer case. Probably need to update how validation data is passed.
def auto_optimise(training_data: np.ndarray | SingleCell,
                  validation_data: np.ndarray | None, true_nn: np.ndarray,
                  num_trials: int, objective: typing.Callable,
                  hyperparam_ranges: dict, algorithm: typing.Callable) -> tuple[
    pl.DataFrame, optuna.Study]:
    # If we are working with Michael's implementation, make sure we pass only
    # the PCs to FAISS.
    # Compile other parameters for the algorithm.
    other_obj_params = {'true_neighbours': true_nn}
    # Minimize runtime and maximize accuracy.
    study = optuna.create_study(directions=['minimize', 'maximize'])

    for _ in range(num_trials):
        study.optimize(
            lambda trial: objective(trial, training_data, validation_data,
                                    hyperparam_ranges, other_obj_params,
                                    algorithm),
            n_trials=1)
        # Don't deepcopy since we want mutation to persist.
        trials = study.get_trials(deepcopy=False)
        last_trial = trials[-1]

        # values_0 is runtime, values_1 is acc.
        # Mark trial failed if it was to slow or very inaccurate or was pruned due to timeout.
        if last_trial.state == optuna.trial.TrialState.PRUNED:
            print(
                f'Trial is a failure because it timed out')
            last_trial.state = optuna.trial.TrialState.FAIL
        elif last_trial.values[0] > MAXIMUM_RUNTIME or last_trial.values[
            1] < MINIMUM_ACCURACY:
            print(
                f'Trial is a failure because runtime was {last_trial.values[0]} and accuracy was {last_trial.values[1]}')
            last_trial.state = optuna.trial.TrialState.FAIL
        else:
            print(
                f'Trial is a success because runtime was {last_trial.values[0]} and accuracy was {last_trial.values[1]}')
            last_trial.state = optuna.trial.TrialState.COMPLETE

    results: pd.DataFrame = study.trials_dataframe()
    # print(results.columns)
    # Every hyperparameter has params_ appended before it by Optuna naming convention for some reason.
    columns_req = ['number', 'values_0',
                   'values_1'] + convert_hyperparam_to_column_names(
        hyperparam_ranges)
    results = results[columns_req]
    return pl.from_pandas(results).rename(
        {'values_0': 'runtime', 'values_1': 'accuracy'}), study


def convert_hyperparam_to_column_names(hyperparam_ranges: dict) -> list:
    return [f'params_{hyperparam}' for hyperparam in hyperparam_ranges]


def save_plots(study, dataset_name: str, algorithm_name: str, size: int):
    # Plot pareto.
    pareto: plotly.graph_objects.Figure = optuna.visualization.plot_pareto_front(
        study,
        target_names=["runtime",
                      "accuracy"])
    pareto.write_image(
        f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_sample_pareto_{NUM_NEIGHBOURS}_nn.png')

    # Plot params corresponding to highest accuracy trials.
    # parallel_accuracy: plotly.graph_objects.Figure = optuna.visualization.plot_parallel_coordinate(
    #     study,
    #     target=lambda targets: targets.values[1], target_name="accuracy")
    #
    # # Reverse direction of scale because it defaults to minimization (even
    # # for accuracy) and apply changes.
    # colorscale = plotly.colors.sequential.Blues[::-1]
    #
    # parallel_accuracy.update_traces(line=dict(colorscale=colorscale))
    #
    # parallel_accuracy.write_image(
    #     f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_sample_parallel_accuracy.png')

    # # Plot param importances.
    # param_importances: plotly.graph_objects.Figure = optuna.visualization.plot_param_importances(
    #     study)
    #
    # param_importances.write_image(
    #     f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_param_importances.png')

    # Plot contours for important parameters.
    candidates_clusters_contour: plotly.graph_objects.Figure = optuna.visualization.plot_contour(
        study, params=['num_candidates_per_neighbour', 'num_clusters'],
        target=lambda targets: targets.values[1], target_name="accuracy")

    candidates_iterations_contour: plotly.graph_objects.Figure = optuna.visualization.plot_contour(
        study, params=['num_candidates_per_neighbour', 'num_kmeans_iterations'],
        target=lambda targets: targets.values[1], target_name="accuracy")

    clusters_iterations_contour: plotly.graph_objects.Figure = optuna.visualization.plot_contour(
        study, params=['num_clusters', 'num_kmeans_iterations'],
        target=lambda targets: targets.values[1], target_name="accuracy")

    candidates_clusters_contour.write_image(
        f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_candidates_clusters_contour_{NUM_NEIGHBOURS}_nn.png')
    candidates_iterations_contour.write_image(
        f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_candidates_iterations_contour_{NUM_NEIGHBOURS}_nn.png')
    clusters_iterations_contour.write_image(
        f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_clusters_iterations_contour_{NUM_NEIGHBOURS}_nn.png')


if __name__ == '__main__':
    # green_sc: SingleCell = load_green_sc(
    #     sc_file=f'{PROJECT_DIR}/single-cell/Green/p400_qced_shareable.h5ad',
    #     retrieve=True, save_to=f'{SCRATCH_DIR}/rosmap')

    ### Auto-optimisation w/ default FAISS IVF algorithm. ###
    # Try a couple of dataset sizes (for Green)...
    # Sample data points.
    # sizes = [1]
    # sampled_green_pcs = []
    # sampled_seadd_pcs = []
    #
    # # Load up datasets.
    # for i in range(len(sizes)):
    #     size = sizes[i]
    #     # First Green...
    #     if not os.path.exists(f'{KNN_DIR}/data/pcs/green_pcs_{size}.npy'):
    #         sampled_cells = SingleCell(
    #             f'{PROJECT_DIR}/single-cell/Subsampled/Green_{size}.h5ad',
    #             X=False)
    #         sampled_pcs = sampled_cells.obsm['PCs']
    #         np.save(f'{KNN_DIR}/data/pcs/green_pcs_{size}', sampled_pcs)
    #         sampled_green_pcs.append(sampled_pcs)
    #     else:
    #         sampled_pcs = np.load(f'{KNN_DIR}/data/pcs/green_pcs_{size}.npy')
    #         sampled_green_pcs.append(sampled_pcs)
    #
    #     # Now SEAAD.
    #     if not os.path.exists(f'{KNN_DIR}/data/pcs/seaad_pcs_{size}.npy'):
    #         sampled_cells = SingleCell(
    #             f'{PROJECT_DIR}/single-cell/Subsampled/SEAAD_{size}.h5ad',
    #             X=False)
    #         sampled_pcs = sampled_cells.obsm['PCs']
    #         np.save(f'{KNN_DIR}/data/pcs/seaad_pcs_{size}', sampled_pcs)
    #         sampled_seadd_pcs.append(sampled_pcs)
    #     else:
    #         sampled_pcs = np.load(f'{KNN_DIR}/data/pcs/seaad_pcs_{size}.npy')
    #         sampled_seadd_pcs.append(sampled_pcs)
    #
    # # Run auto-optimisation framework!
    # for i in range(len(sizes)):
    #     # Get training and validation data.
    #     size, green_data, seaad_data = sizes[i], sampled_green_pcs[i], \
    #         sampled_seadd_pcs[i]
    #     green_train, green_val = split_data(green_data, TRAINING_SPLIT, None,
    #                                         None) if APPLICATION != 'umap' else green_data, green_data
    #     seaad_train, seaad_val = split_data(seaad_data, TRAINING_SPLIT, None,
    #                                         None) if APPLICATION != 'umap' else seaad_data, seaad_data
    #
    #     # Initialise hyperparams.
    #     NUM_CLUSTERS_SEAAD = (1, len(seaad_train) / 39, 'int', True)
    #     NUM_CLUSTERS_GREEN = (1, len(green_train) / 39, 'int', True)
    #     NUM_CLUSTERS_SEARCHED = (1, 100, 'int', True)
    #     SUBSAMPLING_FACTOR = (0, 1, 'float', False)
    #
    #     hyperparam_ranges_faiss_default_green = {
    #         'num_voronoi_cells': NUM_CLUSTERS_SEARCHED,
    #         'num_centroids': NUM_CLUSTERS_GREEN,
    #         'subsampling_factor': SUBSAMPLING_FACTOR}
    #
    #     hyperparam_ranges_faiss_default_seaad = {
    #         'num_voronoi_cells': NUM_CLUSTERS_SEARCHED,
    #         'num_centroids': NUM_CLUSTERS_SEAAD,
    #         'subsampling_factor': SUBSAMPLING_FACTOR}
    #
    #     # First Green.
    #     if not os.path.exists(
    #             f'{KNN_DIR}/faiss/default_green_auto_tuning_{size}_sample_results'):
    #         current_sampled_green_pcs = sampled_green_pcs[i]
    #         results, green_study = auto_optimise(green_train, green_val,
    #                                              NUM_TRIALS, objective_knn,
    #                                              hyperparam_ranges_faiss_default_green,
    #                                              profile_faiss_ivf)
    #         results = results.sort('accuracy', descending=True)
    #         results.write_csv(
    #             f'{KNN_DIR}/faiss/default_green_auto_tuning_{size}_sample_results',
    #             separator='\t')
    #         save_plots(green_study, 'green', 'default', size)
    #
    #     # Now SEAAD.
    #     if not os.path.exists(
    #             f'{KNN_DIR}/faiss/default_seaad_auto_tuning_{size}_sample_results'):
    #         current_sampled_seadd_pcs = sampled_seadd_pcs[i]
    #         results, seaad_study = auto_optimise(seaad_train, seaad_val,
    #                                              NUM_TRIALS, objective_knn,
    #                                              hyperparam_ranges_faiss_default_seaad,
    #                                              profile_faiss_ivf)
    #         results = results.sort('accuracy', descending=True)
    #         results.write_csv(
    #             f'{KNN_DIR}/faiss/default_seaad_auto_tuning_{size}_sample_results',
    #             separator='\t')
    #         save_plots(seaad_study, 'seaad', 'default', size)

    ### Auto-optimisation w/ Michael's implementation. ###
    size = 100
    # Load up datasets.
    # First Green...
    sampled_cells = SingleCell(
        f'{DATA_DIR}/single-cell/Subsampled/Green_{size}.h5ad',
        X=False)
    green_train, green_val = sampled_cells, sampled_cells

    # Now SEAAD.
    sampled_cells = SingleCell(
        f'{DATA_DIR}/single-cell/Subsampled/SEAAD_{size}.h5ad',
        X=False)
    seaad_train, seaad_val = sampled_cells, sampled_cells

    # First, run with default parameters to get baseline performance over a couple of trials.
    green_baseline = {
        "accuracy": [],
        'runtime': [],
        "num_clusters": [],
        "min_clusters_searched": [],
        "max_clusters_searched": [],
        "num_candidates_per_neighbour": [],
        "num_kmeans_iterations": [],
        "kmeans_barbar": [],
    }

    seaad_baseline = {
        "accuracy": [],
        'runtime': [],
        "num_clusters": [],
        "min_clusters_searched": [],
        "max_clusters_searched": [],
        "num_candidates_per_neighbour": [],
        "num_kmeans_iterations": [],
        "kmeans_barbar": [],
    }

    # Get true neighbours and cache if required.
    if not os.path.exists(
            f'{KNN_DIR}/data/seaad_true_{NUM_NEIGHBOURS}_nn_{size}.npy'):
        true_nn_green = brute_force_knn(green_train.obsm['PCs'],
                                        green_val.obsm['PCs'], NUM_NEIGHBOURS,
                                        False, 'faiss',
                                        f'{KNN_DIR}/data/green_true_{NUM_NEIGHBOURS}_nn_{size}.npy',
                                        None, reject_self_neighbours=True)

        true_nn_seaad = brute_force_knn(seaad_train.obsm['PCs'],
                                        seaad_val.obsm['PCs'], NUM_NEIGHBOURS,
                                        False, 'faiss',
                                        f'{KNN_DIR}/data/seaad_true_{NUM_NEIGHBOURS}_nn_{size}.npy',
                                        None, reject_self_neighbours=True)
    else:
        true_nn_green = brute_force_knn(green_train.obsm['PCs'],
                                        green_val.obsm['PCs'], NUM_NEIGHBOURS,
                                        True, 'faiss',
                                        f'{KNN_DIR}/data/green_true_{NUM_NEIGHBOURS}_nn_{size}.npy',
                                        None, reject_self_neighbours=True)

        true_nn_seaad = brute_force_knn(seaad_train.obsm['PCs'],
                                        seaad_val.obsm['PCs'], NUM_NEIGHBOURS,
                                        True, 'faiss',
                                        f'{KNN_DIR}/data/seaad_true_{NUM_NEIGHBOURS}_nn_{size}.npy',
                                        None, reject_self_neighbours=True)

    NUM_BASELINE_TRIALS = 1

    # Compute green defaults.
    num_green_cells = len(green_train.obsm['PCs'])
    num_clusters_green = np.ceil(
        np.minimum(np.sqrt(num_green_cells), num_green_cells / 100)).astype(
        int)
    min_clusters_searched_green = min(10,
                                      NUM_NEIGHBOURS,
                                      num_clusters_green)
    max_clusters_searched_green = min(NUM_NEIGHBOURS,
                                      num_clusters_green)
    num_kmeans_iterations_green, num_kmeans_iterations_seaad = 10, 10
    num_candidates_per_neighbour_green, num_candidates_per_neighbour_seaad = 10, 10
    kmeans_barbar_green, kmeans_barbar_seaad = False, False

    # Compute seaad defaults.
    num_seaad_cells = len(seaad_train.obsm['PCs'])
    num_clusters_seaad = np.ceil(
        np.minimum(np.sqrt(num_seaad_cells), num_seaad_cells / 100)).astype(
        int)
    min_clusters_seaad = min(10, NUM_NEIGHBOURS, num_clusters_seaad)
    max_clusters_seaad = min(NUM_NEIGHBOURS, num_clusters_seaad)

    # Report sizes of datasets.
    print(
        f'Green has {num_green_cells} cells and SEAAD has {num_seaad_cells} cells.')

    for i in range(NUM_BASELINE_TRIALS):
        # First green, then seaad.
        neighbours_green, runtime_green = profile_faiss_michael(green_train,
                                                                None,
                                                                num_clusters=None,
                                                                min_clusters_searched=None,
                                                                max_clusters_searched=None,
                                                                num_candidates_per_neighbour=10,
                                                                num_kmeans_iterations=10,
                                                                num_neighbours=NUM_NEIGHBOURS)

        # Compute accuracy.
        accuracy_green = compute_accuracies(true_nn_green,
                                            neighbours_green,
                                            NUM_NEIGHBOURS)

        # Add to results dict.
        green_baseline["num_clusters"].append(num_clusters_green)
        green_baseline["min_clusters_searched"].append(
            min_clusters_searched_green)
        green_baseline["max_clusters_searched"].append(
            max_clusters_searched_green)

        green_baseline["num_candidates_per_neighbour"].append(
            num_candidates_per_neighbour_green)
        green_baseline["num_kmeans_iterations"].append(
            num_kmeans_iterations_green)
        green_baseline['kmeans_barbar'].append(kmeans_barbar_green)

        green_baseline["runtime"].append(runtime_green)
        green_baseline["accuracy"].append(accuracy_green)

        neighbours_seaad, runtime_seaad = profile_faiss_michael(seaad_train,
                                                                None,
                                                                num_clusters=None,
                                                                min_clusters_searched=None,
                                                                max_clusters_searched=None,
                                                                num_candidates_per_neighbour=10,
                                                                num_kmeans_iterations=10,
                                                                num_neighbours=NUM_NEIGHBOURS)

        accuracy_seaad = compute_accuracies(true_nn_seaad,
                                            neighbours_seaad,
                                            NUM_NEIGHBOURS)

        seaad_baseline["num_clusters"].append(num_clusters_seaad)
        seaad_baseline["min_clusters_searched"].append(min_clusters_seaad)
        seaad_baseline["max_clusters_searched"].append(max_clusters_seaad)

        seaad_baseline["num_candidates_per_neighbour"].append(
            num_candidates_per_neighbour_green)
        seaad_baseline["num_kmeans_iterations"].append(
            num_kmeans_iterations_green)
        seaad_baseline['kmeans_barbar'].append(kmeans_barbar_green)

        seaad_baseline["runtime"].append(runtime_seaad)
        seaad_baseline["accuracy"].append(accuracy_seaad)

    # Convert to polars dataframe and write.
    seaad_baseline_df = pl.DataFrame(seaad_baseline)
    green_baseline_df = pl.DataFrame(green_baseline)
    seaad_baseline_df.write_csv(
        f'{KNN_DIR}/faiss/michael_seaad_{size}_sample_baseline_{NUM_NEIGHBOURS}_nn',
        separator='\t')
    green_baseline_df.write_csv(
        f'{KNN_DIR}/faiss/michael_green_{size}_sample_baseline_{NUM_NEIGHBOURS}_nn',
        separator='\t')

    # Auto-optimise.
    # Initialise hyperparams.
    # 39 value was obtained from FAISS docs - specifies maximum number of
    # clusters suggested.
    # We should obviously split into at least as many clusters as there are neighbours.
    NUM_CLUSTERS_GREEN = (
        len(green_train) / 1500, len(green_train) / 150, 'int', True)
    NUM_CLUSTERS_SEAAD = (
        len(green_train) / 1500, len(seaad_train) / 150, 'int', True)

    # Shouldn't try to search more clusters than there exist clusters.
    # Clearly we should try to search at least as many clusters as there are neighbours.
    MIN_CLUSTERS_SEARCHED = (NUM_NEIGHBOURS, 'num_clusters', 'int', True)
    MAX_CLUSTERS_SEARCHED = (
        'min_clusters_searched', 'num_clusters', 'int', True)

    NUM_CANDIDATES_PER_NEIGHBOUR = (5, 50, 'int', True)
    NUM_KMEANS_ITERATIONS = (1, 10, 'int', True)

    hyperparam_ranges_faiss_default_green = {
        'num_clusters': NUM_CLUSTERS_GREEN,
        'min_clusters_searched': MIN_CLUSTERS_SEARCHED,
        'max_clusters_searched': MAX_CLUSTERS_SEARCHED,
        'num_candidates_per_neighbour': NUM_CANDIDATES_PER_NEIGHBOUR,
        'num_kmeans_iterations': NUM_KMEANS_ITERATIONS,
    }

    hyperparam_ranges_faiss_default_seaad = {
        'num_clusters': NUM_CLUSTERS_SEAAD,
        'min_clusters_searched': MIN_CLUSTERS_SEARCHED,
        'max_clusters_searched': MAX_CLUSTERS_SEARCHED,
        'num_candidates_per_neighbour': NUM_CANDIDATES_PER_NEIGHBOUR,
        'num_kmeans_iterations': NUM_KMEANS_ITERATIONS,
    }
    # First Green.
    results, green_study = auto_optimise(green_train, None,
                                         true_nn_green,
                                         NUM_TRIALS, objective_knn,
                                         hyperparam_ranges_faiss_default_green,
                                         profile_faiss_michael)
    results = results.sort('accuracy', descending=True)
    results.write_csv(
        f'{KNN_DIR}/faiss/michael_green_auto_tuning_{size}_sample_results_{NUM_NEIGHBOURS}_nn',
        separator='\t')
    save_plots(green_study, 'green', 'michael', size)

    # Now SEAAD.
    results, seaad_study = auto_optimise(seaad_train, None,
                                         true_nn_seaad,
                                         NUM_TRIALS, objective_knn,
                                         hyperparam_ranges_faiss_default_seaad,
                                         profile_faiss_michael)
    results = results.sort('accuracy', descending=True)
    results.write_csv(
        f'{KNN_DIR}/faiss/michael_seaad_auto_tuning_{size}_sample_results_{NUM_NEIGHBOURS}_nn',
        separator='\t')
    save_plots(seaad_study, 'seaad', 'michael', size)

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

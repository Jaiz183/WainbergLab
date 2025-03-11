import os
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

NUM_NEIGHBOURS = 5
APPLICATION = 'umap'
TRAINING_SPLIT = 70
NUM_TRIALS = 2


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
    _, _, accuracy = compute_accuracies(true_neighbours, neighbours, 5)

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
        if param_type == 'int':
            current_hyperparams[hyperparam] = trial.suggest_int(
                hyperparam, min_val,
                max_val,
                log=is_log)
        else:
            current_hyperparams[hyperparam] = trial.suggest_float(
                hyperparam, min_val,
                max_val,
                log=is_log)

    # Add on non-hyperparameters
    current_hyperparams['training_data'], current_hyperparams[
        'validation_data'], current_hyperparams[
        'num_neighbours'] = training_dataset, validation_dataset, NUM_NEIGHBOURS
    # Compute objective values.
    neighbours, runtime = algorithm(
        **current_hyperparams)
    _, _, accuracy = compute_accuracies(true_neighbours, neighbours,
                                        NUM_NEIGHBOURS)

    return float(runtime), float(accuracy)


# TODO: update function after updating Michael's implementation to handle label
#  transfer case. Probably need to update how validation data is passed.
def auto_optimise(training_data: np.ndarray | SingleCell,
                  validation_data: np.ndarray | SingleCell,
                  num_trials: int, objective: typing.Callable,
                  hyperparam_ranges: dict, algorithm: typing.Callable) -> tuple[
    pl.DataFrame, optuna.Study]:
    # If we are working with Michael's implementation, make sure we pass only
    # the PCs to FAISS.
    # TODO: validation data is not in correct format.
    if isinstance(training_data, SingleCell):
        true_nn = brute_force_knn(training_data.obsm['PCs'],
                                  validation_data.obsm['PCs'], NUM_NEIGHBOURS,
                                  False, 'faiss',
                                  None, None, reject_self_neighbours=True)
    else:
        true_nn = brute_force_knn(training_data, validation_data,
                                  NUM_NEIGHBOURS,
                                  False, 'faiss',
                                  None, None, reject_self_neighbours=False)
    # Compile other parameters for the algorithm.
    other_obj_params = {'true_neighbours': true_nn}
    # Minimize runtime and maximize accuracy.
    study = optuna.create_study(directions=['minimize', 'maximize'])
    study.optimize(
        lambda trial: objective(trial, training_data, validation_data,
                                hyperparam_ranges, other_obj_params, algorithm),
        n_trials=num_trials)

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
        f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_sample_pareto.png')

    # Plot params corresponding to highest accuracy trials.
    parallel_accuracy: plotly.graph_objects.Figure = optuna.visualization.plot_parallel_coordinate(
        study,
        target=lambda targets: targets.values[1], target_name="accuracy")

    parallel_accuracy.write_image(
        f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_sample_parallel_accuracy.png')

    # Plot param importances.
    param_importances: plotly.graph_objects.Figure = optuna.visualization.plot_param_importances(
        study)

    param_importances.write_image(
        f'{KNN_DIR}/faiss/{algorithm_name}_{dataset_name}_auto_tuning_{size}_param_importances.png')


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
    sizes = [1]
    green_subsamples, seaad_subsamples = [], []

    # Load up datasets.
    for i in range(len(sizes)):
        size = sizes[i]
        # First Green...
        sampled_cells = SingleCell(
            f'{PROJECT_DIR}/single-cell/Subsampled/Green_{size}.h5ad',
            X=False)
        green_subsamples.append(sampled_cells)

        # Now SEAAD.
        sampled_cells = SingleCell(
            f'{PROJECT_DIR}/single-cell/Subsampled/SEAAD_{size}.h5ad',
            X=False)
        seaad_subsamples.append(sampled_cells)

    # Auto-optimise.
    for i in range(len(sizes)):
        size = sizes[i]
        green_data, seaad_data = green_subsamples[i], seaad_subsamples[i]
        # TODO: split up data after generalising Michael's function to
        #  label transfer case.
        green_train, green_val = green_data, green_data
        seaad_train, seaad_val = seaad_data, seaad_data

        # TODO: Decide on hyperparam ranges.
        # Initialise hyperparams.
        # 39 value was obtained from FAISS docs - specifies maximum number of
        # clusters suggested.
        # We should obviously split into at least as many clusters as there are neighbours.
        NUM_CLUSTERS_GREEN = (
        NUM_NEIGHBOURS, len(green_train) / 39, 'int', True)
        NUM_CLUSTERS_SEAAD = (
        NUM_NEIGHBOURS, len(seaad_train) / 39, 'int', True)

        # Shouldn't try to search more clusters than there exist clusters.
        # Clearly we should try to search at least as many clusters as there are neighbours.
        MIN_CLUSTERS_SEARCHED = (NUM_NEIGHBOURS, 'num_clusters', 'int', True)
        MAX_CLUSTERS_SEARCHED = (
        'min_clusters_searched', 'num_clusters', 'int', True)

        NUM_CANDIDATES_PER_NEIGHBOUR = (5, 100, 'int', True)
        NUM_KMEANS_ITERATIONS = (1, 10, 'int', True)
        KMEANS_BARBAR = (0, 1, 'int', False)

        hyperparam_ranges_faiss_default_green = {
            'num_clusters': NUM_CLUSTERS_GREEN,
            'min_clusters_searched': MIN_CLUSTERS_SEARCHED,
            'max_clusters_searched': MAX_CLUSTERS_SEARCHED,
            'num_candidates_per_neighbour': NUM_CANDIDATES_PER_NEIGHBOUR,
            'num_kmeans_iterations': NUM_KMEANS_ITERATIONS,
            'kmeans_barbar': KMEANS_BARBAR}

        hyperparam_ranges_faiss_default_seaad = {
            'num_clusters': NUM_CLUSTERS_SEAAD,
            'min_clusters_searched': MIN_CLUSTERS_SEARCHED,
            'max_clusters_searched': MAX_CLUSTERS_SEARCHED,
            'num_candidates_per_neighbour': NUM_CANDIDATES_PER_NEIGHBOUR,
            'num_kmeans_iterations': NUM_KMEANS_ITERATIONS,
            'kmeans_barbar': KMEANS_BARBAR}

        # First Green.
        if not os.path.exists(
                f'{KNN_DIR}/faiss/michael_green_auto_tuning_{size}_sample_results'):
            current_sampled_green_pcs = green_subsamples[i]
        results, green_study = auto_optimise(green_train, green_val,
                                             NUM_TRIALS, objective_knn,
                                             hyperparam_ranges_faiss_default_green,
                                             profile_faiss_michael)
        results = results.sort('accuracy', descending=True)
        results.write_csv(
            f'{KNN_DIR}/faiss/michael_green_auto_tuning_{size}_sample_results',
            separator='\t')
        save_plots(green_study, 'green', 'michael', size)

        # Now SEAAD.
        if not os.path.exists(
                f'{KNN_DIR}/faiss/michael_seaad_auto_tuning_{size}_sample_results'):
            current_sampled_seadd_pcs = seaad_subsamples[i]
        results, seaad_study = auto_optimise(seaad_train, seaad_val,
                                             NUM_TRIALS, objective_knn,
                                             hyperparam_ranges_faiss_default_seaad,
                                             profile_faiss_michael)
        results = results.sort('accuracy', descending=True)
        results.write_csv(
            f'{KNN_DIR}/faiss/michael_seaad_auto_tuning_{size}_sample_results',
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

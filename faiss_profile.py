from knn_profile import profile_faiss_ivf, load_pcs, sample_data, \
    compound_trials
import numpy as np

faiss_ivf_hyperparameters = {}

algorithm_parameters = {
    'faiss_ivf': (profile_faiss_ivf, faiss_ivf_hyperparameters),
}


def get_num_centroids(num_cells: int) -> int:
    return np.ceil(min(np.sqrt(num_cells), num_cells / 100))


if __name__ == '__main__':
    pcs = load_pcs()
    sample, sample_indices, sample_train_data, sample_test_data = sample_data(
        pcs, 10)
    compound_trials()

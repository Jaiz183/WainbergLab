import os.path
import pickle

import numpy
import numpy as np

from ryp import r, to_py, to_r
from single_cell import SingleCell
from constants import *
from utils import run
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import polars as pl

from binarytree import Node

from scipy.cluster.hierarchy import fcluster, cut_tree, is_valid_linkage


# IDEA - init. a tree, label via cluster size, find number of leaves recursively after.
# Otherwise, you'd have to go up the tree per leaf (point), which would be slow... - might be useful later too.
# I don't think we need a tree - can just recursively find the children's, it'd just take O(n) for search.

# TODO: update to correct IDs.
def get_dendrogram(dataframe: pl.DataFrame, root: int, num_cells: int) -> Node:
    """
    :param num_cells: number of cells.
    :param dataframe: merge field of hclust object from HGC or any dataframe
    whose columns are the subclusters / points merged and whose rows are
    subclusters formed. Points should be indexed -num_cells to -1 and
    subclusters 1 to ...
    :param root: largest subcluster, usually has val == len(dataframe).
    :return:
    """
    # Init root.
    root_node = Node(root)

    # Is the largest cluster, handle separately.

    # Is a subcluster. Has children so find them rec.
    # Recall that r indexing starts at one, so we should sub. one.
    if root > 0:
        left_root = int(dataframe['column_0'][root - 1])
        right_root = int(dataframe['column_1'][root - 1])
        left = get_dendrogram(dataframe, left_root, num_cells)
        right = get_dendrogram(dataframe, right_root, num_cells)
        root_node.left, root_node.right = left, right

    return root_node


def get_subcluster_sizes(root: Node, sizes: dict[int, int]):
    """
    Computes size of every subcluster w/ Pythonic notation, that is, 0 - n - 1
    for the points and n+ for clusters.
    :param root: root of dendrogram
    :param sizes: dictionary to store dendrogram sizes in.
    :return:
    """
    # Leaf base case.
    if root.left is None and root.right is None:
        sizes[root.val] = 1
    else:
        # If not leaf, must have exactly two children because of agglomerative
        # clustering.
        # Update left and right subs.
        get_subcluster_sizes(root.left, sizes)
        get_subcluster_sizes(root.right, sizes)
        # First condition is technically redundant because we should only ever
        # update root.val once, at which point we insert into the
        # dict and set it to the correct size.
        sizes[root.val] = sizes[root.left.val] + \
                          sizes[root.right.val]


if __name__ == '__main__':
    run('export LANG=en_US.UTF-8')
    run('export LC_ALL=en_US.UTF-8')
    r('devtools::install_github("XuegongLab/HGC")')
    # Set up dendextend.

    # Cluster.
    SIZE = 10
    NUM_CLUSTERS_CUTREE, NUM_CLUSTERS_CUT_TREE, NUM_CLUSTERS_FCLUST = 15, 15, 15
    sc = (SingleCell(f'{KNN_DIR}/data/Green_{SIZE}.h5ad')
          .set_var_names('_index'))

    snn_graph = sc.obsp['shared_neighbors']

    if not os.path.exists(f'{CLUSTER_DIR}/data/snn_graph_green_{SIZE}'):
        to_r(snn_graph.T, 'snn_graph')
        r(f'saveRDS(snn_graph, "{CLUSTER_DIR}/data/snn_graph_green_{SIZE}")')
    else:
        r(f'snn_graph = readRDS("{CLUSTER_DIR}/data/snn_graph_green_{SIZE}")')

    r('clusters = HGC::HGC.dendrogram(snn_graph)')
    r('dendrogram = as.dendrogram(clusters)')

    ### Try cutree in R. ###
    r(f'cluster_assignments = cutree(clusters, k={NUM_CLUSTERS_CUTREE})')

    # Returns a data frame with index and cluster label columns.
    cluster_assignments_cutree = to_py('cluster_assignments')

    ### Try cut_tree in Python. ###
    # Retrieve object and cache.
    if not os.path.exists(f'{CLUSTER_DIR}/data/hclust_green_{SIZE}'):
        hclust = to_py('clusters')
        with open(f'{CLUSTER_DIR}/data/hclust_green_{SIZE}', 'wb') as f:
            pickle.dump(hclust, f)
    else:
        with open(f'{CLUSTER_DIR}/data/hclust_green_{SIZE}', 'rb') as f:
            hclust = pickle.load(f)

    # Important attributes are - 'merge', 'height'.
    num_cells = len(sc.obsm['PCs'])

    merge: pl.DataFrame = hclust['merge']
    num_subclusters = len(merge)

    # Get dendrogram and subcluster sizes.
    dendrogram = get_dendrogram(merge, root=num_subclusters,
                                num_cells=num_cells)

    subcluster_sizes = {}
    get_subcluster_sizes(dendrogram, subcluster_sizes)

    # Leave only true subclusters. Remove leaves.
    subcluster_sizes = {key: val for key, val in subcluster_sizes.items() if
                        key > 0}

    # Transform labels to positive 0 -> n - 1 for points (< 0 in hclust), and n+ for subclust. (1+ in hclust).
    merge = (merge.with_columns(column_0=
                                pl.when(pl.col('column_0') < 0)
                                .then(-pl.col('column_0').cast(pl.Int32) - 1)
                                .otherwise(pl.col('column_0').cast(
                                    pl.Int32) + num_cells - 1),
                                column_1=
                                pl.when(pl.col('column_1') < 0)
                                .then(-pl.col('column_1').cast(pl.Int32) - 1)
                                .otherwise(pl.col('column_1').cast(
                                    pl.Int32) + num_cells - 1)))

    # Create from mapping.
    # Keep only subclusters, disregard points.
    subcluster_sizes_reformatted = {'subcluster': [], 'sizes': []}
    for subcluster, size in subcluster_sizes.items():
        subcluster_sizes_reformatted['subcluster'].append(subcluster)
        subcluster_sizes_reformatted['sizes'].append(size)

    subcluster_sizes_df = pl.DataFrame(subcluster_sizes_reformatted).sort(
        by='subcluster').drop('subcluster')

    cluster_df = pl.concat(
        [merge, pl.DataFrame(hclust['height']), subcluster_sizes_df],
        how='horizontal')

    # Checking for bugs.
    # test: pl.DataFrame = cluster_df.group_by('column_1').agg(pl.col('column_0').count())
    # print(test.filter(pl.col('column_0') != 1))
    # print(cluster_df.filter(pl.col('column_0') == pl.col('column_1')))
    # print(cluster_df.filter((pl.col('column_0') == 0).or_(pl.col('column_1') == 0)))

    # Convert to numpy.
    # Cutree.
    cluster_assignments_cutree = np.array(
        cluster_assignments_cutree).flatten()
    cluster_assignments_cutree = np.array(
        [x for x in cluster_assignments_cutree if x is not None])
    # print(cluster_assignments_cutree)

    # Cut_tree.
    cluster_arr = cluster_df.to_numpy()
    assert is_valid_linkage(cluster_arr)
    cluster_assignments_cut_tree = cut_tree(cluster_arr, NUM_CLUSTERS_CUT_TREE)
    cluster_assignments_cut_tree = np.array(
        cluster_assignments_cut_tree).flatten()

    # fclust.
    cluster_assignments_fclust = fcluster(cluster_arr, NUM_CLUSTERS_FCLUST, 'maxclust')
    cluster_assignments_fclust = np.array(
        cluster_assignments_fclust).flatten()

    sc.obs = (sc.obs.with_columns(
        cut_tree_clusters=cluster_assignments_cut_tree,
        cutree_clusters=cluster_assignments_cutree,
        fclust_clusters=cluster_assignments_fclust)
              .with_columns(pl.col('cut_tree_clusters').cast(pl.String),
                            pl.col('cutree_clusters').cast(pl.String),
                            pl.col('fclust_clusters').cast(pl.String)))

    # none_params = {'lightness_range': None, 'chroma_range': None, 'hue_range': None,
    # 'first_color': None, 'stride': None}
    none_params = {}

    sc.plot_embedding('cut_tree_clusters',
                      filename=f'{CLUSTER_DIR}/results/pacmap_cut_tree_green_{SIZE}.png',
                      **none_params)
    sc.plot_embedding('cutree_clusters',
                      filename=f'{CLUSTER_DIR}/results/pacmap_cutree_green_{SIZE}.png',
                      **none_params)
    sc.plot_embedding('fclust_clusters',
                      filename=f'{CLUSTER_DIR}/results/pacmap_fclust_green_{SIZE}.png',
                      **none_params)

import os.path

from ryp import r, to_py, to_r
from single_cell import SingleCell
from constants import *
from utils import run

import matplotlib.pyplot as plt
run('export LANG=en_US.UTF-8')
run('export LC_ALL=en_US.UTF-8')
r('devtools::install_github("XuegongLab/HGC")')
# Set up dendextend.
# r('install.packages(\'dendextend\')')

sc = (SingleCell(f'{KNN_DIR}/data/Green_1.h5ad')
            .set_var_names('_index'))

print(sc.obsm['PaCMAP'])

snn_graph = sc.obsp['shared_neighbors']
to_r(snn_graph.T, 'snn_graph')
r('clusters = HGC::HGC.dendrogram(snn_graph)')
r('dendrogram = as.dendrogram(clusters)')

# Trying cutree.
r('cluster_assignments = cutree(clusters, k=35)')
# Cover to factor for easier handling.
r('cluster_assignments_fac = as.factor(cluster_assignments)')

# Returns a data frame with index and cluster label columns.
cluster_assignments = to_py('cluster_assignments')

# TODO: plot UMAPs / PaCMAPs.
fig, axs = plt.subplots()
axs.scatter(cluster_assignments[:, 0], cluster_assignments[:, 1])
plt.savefig(f'{CLUSTER_DIR}/results/pacmap.png')

# Plot.
# TODO: fix xlim issue with 10% sample.
r(f'png(filename=\'{CLUSTER_DIR}/results/dendrogram.png\')')
r('clusters$height = log(clusters$height + 1)')
r('HGC::HGC.PlotDendrogram(tree = clusters, k = 5, plot.label = FALSE)')
r('dev.off()')
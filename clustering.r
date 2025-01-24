# Installing packages.
library(glue)
devtools::install_github("XuegongLab/HGC")


# System constants
SCRATCH_DIR = if (SYSTEM == 'narval') 'scratch' else '/scratch/w/wainberg'
PROJECT_DIR = if (SYSTEM == 'narval') 'projects/def-wainberg' else SCRATCH_DIR
HOME_DIR = glue('{PROJECT_DIR}/jeesonja')
CONTROLLED_ACCESS = 'projects/rrg-wainberg/CommonMind/ControlledAccess'
CLUSTER_DIR = glue('{HOME_DIR}/clustering')
KNN_DIR = glue('{HOME_DIR}/knn')

# Need to somehow compute subcluster sizes.
snn_graph = readRDS(glue("{CLUSTER_DIR}/data/snn_graph_green_{SIZE}"))
clusters = HGC::HGC.dendrogram(snn_graph)
dendrogram = as.dendrogram(clusters)


# Get dendrogram.
png(filename=glue('{CLUSTER_DIR}/results/dendrogram.png'))
clusters$height = log(clusters$height + 1)
HGC::HGC.PlotDendrogram(tree = clusters, k = 5, plot.label = FALSE)
dev.off()



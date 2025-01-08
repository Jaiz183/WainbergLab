# Load necessary libraries
library(stats)
library(ggplot2)
library(ggrepel)
library(ggdendro)
library(dplyr)
library(dynamicTreeCut)
library(RANN)
library(Matrix)
library(Seurat)
library(Rcpp)
library(HGC)
library(hdf5r)

# Open the HDF5 file
h5_file <- H5File$new("projects/def-wainberg/single-cell/Subsampled/Green_10.h5ad", mode = "r")

print("Stage 1: Getting raw counts")

# Get raw counts from X
raw_counts_group <- h5_file[["X"]]
counts_data <- raw_counts_group[["data"]]$read()
counts_indptr <- raw_counts_group[["indptr"]]$read()
counts_indices <- raw_counts_group[["indices"]]$read()

# Extract cell identifiers from the obs/_index dataset
cell_ids <- h5_file[["obs"]][["_index"]]$read()

# Create raw counts matrix
n_rows <- length(counts_indptr) - 1
n_cols <- max(counts_indices) + 1
raw_counts <- sparseMatrix(
  i = rep(1:n_rows, diff(counts_indptr)),
  j = counts_indices + 1,
  x = as.numeric(counts_data),
  dims = c(n_rows, n_cols)
)

# Set row and column names for raw_counts
rownames(raw_counts) <- cell_ids
colnames(raw_counts) <- paste0("Gene_", 1:n_cols)  # Adjust if you have gene names

print("Stage 2: Getting shared neighbors")

# Get shared nearest neighbors using the correct path
sn <- h5_file[["obsp/shared_neighbors"]]
n_cells <- length(sn[["indptr"]]$read()) - 1

# Create shared_neighbors matrix
shared_neighbors <- sparseMatrix(
  i = rep(1:n_cells, diff(sn[["indptr"]]$read())),
  j = sn[["indices"]]$read() + 1,
  x = as.numeric(sn[["data"]]$read()),
  dims = c(n_cells, n_cells),
  dimnames = list(cell_ids, cell_ids)  # Add row and column names
)
pcs <- h5_file[["obsm/PCs"]]$read()
# Close the file
h5_file$close()

# Stage 3: Getting the PCs and mapping with SHC
print("Stage 3: Beginning the SHC on Seurat PCs")

# Run SHC on the extracted PCs matrix
hc_dat <- HGC.dendrogram(shared_neighbors)

# clusters <- cutreeDynamicTree(
#   dendro = hc_dat,  # Use the hclust object
#   maxTreeHeight = Inf,  # Set the maximum tree height to infinity
#   minModuleSize = 1,  # Minimum module size is 1
#   deepSplit = FALSE  # Disable deep splitting
# )

clusters <- cutree(hc_dat, k = 20)

# Convert clusters to a factor
clusters <- as.factor(clusters)

# Count the number of unique clusters
num_clusters <- length(unique(clusters))
print(paste("Number of clusters identified:", num_clusters))


# Create Seurat object using the PCs
# Transpose the pcs matrix so cells are rows and PCs are columns
pcs_transposed <- t(pcs)
raw_counts_transposed <- t(raw_counts)

seurat_object <- CreateSeuratObject(counts = raw_counts_transposed)
seurat_object@meta.data$cell_ids <- cell_ids

# Add SHC clusters to the Seurat object metadata
clusters <- as.factor(clusters)  # Convert clusters to a factor
seurat_object@meta.data$shc_clusters <- clusters

# Add the PCs to the Seurat object as a reduction
rownames(pcs_transposed) <- cell_ids  # Set cell IDs as row names
colnames(pcs_transposed) <- paste0("PC_", 1:ncol(pcs_transposed))  # Set PC names as column names

# Now create the DimReduc object and add it to the Seurat object
seurat_object[["pca"]] <- CreateDimReducObject(embeddings = pcs_transposed, key = "PC_", assay = DefaultAssay(seurat_object))
# Run UMAP on the Seurat object using the PCs
seurat_object <- RunUMAP(seurat_object, reduction = "pca", dims = 1:50)

# Function to generate a random hexadecimal color
generate_random_color <- function() {
  sprintf("#%02X%02X%02X", sample(0:255, 1), sample(0:255, 1), sample(0:255, 1))
}
# Generate random colors for each cluster (num_colors should be the number of unique clusters)
num_colors <- length(unique(clusters))  # Assuming clusters is a vector of cluster assignments
random_colors <- replicate(num_colors, generate_random_color())
# Now, create your UMAP plot with the random colors
umap_plot <- DimPlot(seurat_object, reduction = "umap", group.by = "shc_clusters", cols = random_colors)

# Now add labels and theme separately
umap_plot <- umap_plot + labs(title = "UMAP Plot with Cutree Clusters") + theme_minimal()

# Add cluster labels to the UMAP plot
umap_plot <- LabelClusters(plot = umap_plot, id = "shc_clusters", repel = TRUE)
# Save the plot with random colors
png("cutree_hacked_green_10.png", 1200, 800)
print(umap_plot)
dev.off()
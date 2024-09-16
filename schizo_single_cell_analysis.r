library(ACTIONet)
# library(BiocManager)
input_path = './projects/def-wainberg/single-cell/SZBDMulticohort'
sc = readRDS(file.path(input_path, "combinedCells_ACTIONet.rds"))
saveRDS(rownames(sc), "./var")
saveRDS(counts(sc), "./X")
saveRDS(sc, "./sce")

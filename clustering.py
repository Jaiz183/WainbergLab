from ryp import r, to_py, to_r
from single_cell import SingleCell

r('devtools::install_github("XuegongLab/HGC")')

sc = SingleCell('Sst_Chodl.h5ad').set_var_names('feature_name').qc(
    allow_float=True).hvg(allow_float=True).normalize(
    allow_float=True).PCA().neighbors().shared_neighbors()
snn_graph = sc.obsp['shared_neighbors']
to_r(snn_graph.T, 'snn_graph')
r('dendrogram = HGC::HGC.dendrogram(snn_graph)')

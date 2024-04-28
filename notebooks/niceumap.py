#niceumap

import scanpy as sc
import datamapplot
from anndata.experimental import concat_on_disk
from umap import UMAP

#l = [
#  "../data/step_0_predict_part_0_0.h5ad",
#  "../data/step_0_predict_part_0_1.h5ad",
#  "../data/step_0_predict_part_0_2.h5ad",
#  "../data/step_0_predict_part_0_3.h5ad",
#  "../data/step_0_predict_part_1_0.h5ad",
#  "../data/step_0_predict_part_1_1.h5ad",
#  "../data/step_0_predict_part_1_2.h5ad",
#  "../data/step_0_predict_part_1_3.h5ad",
#  "../data/step_0_predict_part_2_0.h5ad",
#  "../data/step_0_predict_part_2_1.h5ad",
#  "../data/step_0_predict_part_2_2.h5ad",
#  "../data/step_0_predict_part_2_3.h5ad",  
#  "../data/step_0_predict_part_3_0.h5ad",
#  "../data/step_0_predict_part_3_1.h5ad",
#  "../data/step_0_predict_part_3_2.h5ad",
#  "../data/step_0_predict_part_3_3.h5ad",
#]
#concat_on_disk(l, "../data/step_0_predict.h5ad", uns_merge="same", index_unique="_")
adata = sc.read_h5ad("data/step_0_predict.h5ad")

fit = UMAP(n_neighbors=15, min_dist=0.1, spread=1)
adata.obsm["X_umap"] = fit.fit_transform(adata.X)


adata.write("data/step_0_predict.h5ad")
fig, ax = datamapplot.create_plot(adata.obsm['X_umap'], 
                                adata.obs['conv_pred_cell_type_ontology_term_id'], 
                                darkmode=True, 
                                title="Predicted Cell Types")

fig.savefig("data/step_0_predict_umap_celltype.png", bbox_inches="tight")
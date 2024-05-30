#niceumap

import scanpy as sc
import datamapplot
from anndata.experimental import concat_on_disk
from umap import UMAP
import os

# srun -p gpu -q gpu --gres=gpu:A40:1,gmem:40G --cpus-per-task 16 --mem-per-gpu 32G --ntasks-per-node=1 scprint predict --config config/base.yml --config config/pretrain_medium.yml --config config/predict.yml --ckpt_path /pasteur/zeus/projets/p02/ml4ig_hot/Users/jkalfon/scprint_scale/vbd8bavn/checkpoints/epoch=17-step=90000.ckpt
# srun -p ml4ig -res=cpu:32,mem:90G --cpus-per-task=32 python niceumap.py

# Assuming the data files are in the "../data/" directory
data_directory = "../data/"
file_list = os.listdir(data_directory)
# Filter out only the files with the 'h5ad' extension
h5ad_files = [file for file in file_list if file.endswith('.h5ad') and "step_0_predict_part" in file]
# Sort the files to maintain order
h5ad_files.sort()
# Update the list 'l' with the full paths of the 'h5ad' files
h5ad_files = [os.path.join(data_directory, file) for file in h5ad_files]


concat_on_disk(h5ad_files, "../data/step_0_predict.h5ad", uns_merge="same", index_unique="_")
adata = sc.read_h5ad("data/step_0_predict.h5ad")

fit = UMAP(n_neighbors=15, min_dist=0.1, spread=1)
adata.obsm["X_umap"] = fit.fit_transform(adata.X)


adata.write("data/step_0_predict.h5ad")
fig, ax = datamapplot.create_plot(adata.obsm['X_umap'], 
                                adata.obs['conv_pred_cell_type_ontology_term_id'], 
                                darkmode=False, 
                                title="Predicted Cell Types")

fig.savefig("data/step_0_predict_umap_celltype.png", bbox_inches="tight")
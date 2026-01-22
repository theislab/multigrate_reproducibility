import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import os
import time
import json
import time
import tracemalloc

from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
from torch.cuda import is_available

from sciPENN.sciPENN_API import sciPENN_API

tracemalloc.start()

mod1_path = snakemake.input.mod1
mod2_path = snakemake.input.mod2
out_path = snakemake.output.h5ad
log_path = snakemake.output.log

mod1_hvf = pd.read_csv(snakemake.input.mod1_hvf, header=None)[0].tolist()
mod2_hvf = pd.read_csv(snakemake.input.mod2_hvf, header=None)[0].tolist()

# data loading and pp
adata_gene = sc.read(mod1_path)
adata_protein = sc.read(mod2_path)

adata_gene = adata_gene[:, mod1_hvf].copy()
adata_protein = adata_protein[:, mod2_hvf].copy()

# adata_gene.X = adata_gene.layers['counts'].copy()
# adata_protein.X = adata_protein.layers['counts'].copy()

del adata_gene.layers
del adata_protein.layers

train_bool = adata_gene.obs['split'] == 'train'

adata_gene_train = adata_gene[train_bool].copy()
adata_protein_train = adata_protein[train_bool].copy()
adata_gene_test = adata_gene[np.invert(train_bool)].copy()
adata_protein_test = adata_protein[np.invert(train_bool)].copy()

sciPENN = sciPENN_API(gene_trainsets = [adata_gene_train], protein_trainsets = [adata_protein_train], 
                      gene_test = adata_gene_test, train_batchkeys = ['Site'], test_batchkey = 'Site', 
                      type_key = 'cell_type', min_genes=False, min_cells = None, select_hvg = False,
                      cell_normalize=False, log_normalize=False)

# to make sure test gene and protein data is normalized the same way as train data
sciPENN_test = sciPENN_API(gene_trainsets = [adata_gene_test], protein_trainsets = [adata_protein_test], 
                      gene_test = None, train_batchkeys = ['Site'], test_batchkey = 'Site', 
                      type_key = 'cell_type', min_genes=False, min_cells = None, select_hvg = False,
                      cell_normalize=False, log_normalize=False)

# train model
start_time = time.time()

weights_dir = out_path.replace('.h5ad', '')

sciPENN.train(quantiles = [0.1, 0.25, 0.75, 0.9], n_epochs = 10000, ES_max = 12, decay_max = 6, 
             decay_step = 0.1, lr = 10**(-3), weights_dir = weights_dir, load = False)

end_time = time.time()

predicted_test = sciPENN.predict()
adata_test_pred = ad.AnnData(X=predicted_test)
adata_test_pred.write_h5ad(out_path)

adata_test_true = ad.AnnData(X=sciPENN_test.proteins)
adata_test_true.write_h5ad(out_path.replace("imputed.h5ad", "true.h5ad"))

elapsed = end_time - start_time

_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

runtime_data = {
    "method": os.path.basename(__file__),
    "time_sec": round(elapsed, 2),
    "peak_memory_mb": round(peak / (1024 ** 2), 2),
}

with open(log_path, "w") as f:
    json.dump(runtime_data, f, indent=2)

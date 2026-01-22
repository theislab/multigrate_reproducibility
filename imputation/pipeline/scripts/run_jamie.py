import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import os
import time
import json
import time
import tracemalloc

import jamie
from jamie import JAMIE

tracemalloc.start()

mod1_path = snakemake.input.mod1
mod2_path = snakemake.input.mod2
out_path = snakemake.output.h5ad
log_path = snakemake.output.log

# data loading and pp
RNA_data = sc.read(mod1_path)
ATAC_data = sc.read(mod2_path)

mod1_hvf = pd.read_csv(snakemake.input.mod1_hvf, header=None)[0].tolist()
mod2_hvf = pd.read_csv(snakemake.input.mod2_hvf, header=None)[0].tolist()

RNA_data = RNA_data[:, mod1_hvf].copy()
ATAC_data = ATAC_data[:, mod2_hvf].copy()

# RNA_data.X = RNA_data.layers['counts'].copy()
# ATAC_data.X = ATAC_data.layers['counts'].copy()

del RNA_data.layers
del ATAC_data.layers

# sc.pp.normalize_total(RNA_data, target_sum=1e4)
# sc.pp.log1p(RNA_data)
# sc.pp.highly_variable_genes(RNA_data, n_top_genes=2000, subset=True)
sc.pp.scale(RNA_data)

# sc.pp.normalize_total(ATAC_data, target_sum=1e4)
# sc.pp.log1p(ATAC_data)
# sc.pp.highly_variable_genes(ATAC_data, n_top_genes=20000, subset=True)
sc.pp.scale(ATAC_data)

RNA_train = RNA_data[RNA_data.obs['split'] == 'train'].copy()
ATAC_train = ATAC_data[ATAC_data.obs['split'] == 'train'].copy()
RNA_test = RNA_data[RNA_data.obs['split'] == 'test'].copy()
ATAC_test = ATAC_data[ATAC_data.obs['split'] == 'test'].copy()

data1 = RNA_train.X.copy()
data2 = ATAC_train.X.copy()
data1_test = RNA_test.X.copy()
data2_test = ATAC_test.X.copy()

del RNA_data
del ATAC_data
del RNA_train
del ATAC_train
del RNA_test
del ATAC_test

corr = np.eye(data1.shape[0], data2.shape[0])

jm = JAMIE(min_epochs=500, device='cuda', use_f_tilde=False)

# train model
start_time = time.time()

integrated_data = jm.fit_transform(dataset=[data1, data2], P=corr)

end_time = time.time()

model_dir = out_path.replace("imputed.h5ad", "")
os.makedirs(model_dir, exist_ok=True)
jm.save_model(out_path.replace("imputed.h5ad", "model.h5"))

new_integrated_data = jm.transform(dataset=[data1_test, np.zeros_like(data2_test)])

adata_test_pred = ad.AnnData(X=jm.modal_predict(data1_test, 0))
adata_test_pred.write_h5ad(out_path)

adata_test_true = ad.AnnData(X=data2_test)
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

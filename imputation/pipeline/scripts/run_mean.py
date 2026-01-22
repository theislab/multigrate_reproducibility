import anndata as ad
import numpy as np
import pandas as pd
import os
import time
import json
import time
import tracemalloc
from utils import load_data

tracemalloc.start()

mod1_path = snakemake.input.mod1
mod2_path = snakemake.input.mod2
out_path = snakemake.output.h5ad
log_path = snakemake.output.log

mod1_hvf = pd.read_csv(snakemake.input.mod1_hvf, header=None)[0].tolist()
mod2_hvf = pd.read_csv(snakemake.input.mod2_hvf, header=None)[0].tolist()

X_input, Y_target, split, cell_types = load_data(mod1_path, mod2_path, mod1_hvf, mod2_hvf)

X_train, X_test = X_input[split == 'train'], X_input[split == 'test']
Y_train, Y_test = Y_target[split == 'train'], Y_target[split == 'test']
cell_types_train, cell_types_test = cell_types[split == 'train'], cell_types[split == 'test']

start_time = time.time()
mean_by_cell_type = {}
unique_cell_types = np.unique(cell_types_train)

for cell_type in unique_cell_types:
    mean_by_cell_type[cell_type] = Y_train[cell_types_train == cell_type].mean(axis=0)

Y_train_pred = np.array([mean_by_cell_type[ct] for ct in cell_types_train])
Y_test_pred = np.array([mean_by_cell_type.get(ct, Y_train.mean(axis=0)) for ct in cell_types_test])

end_time = time.time()
adata_test_pred = ad.AnnData(X=Y_test_pred)
adata_test_pred.write_h5ad(out_path)

adata_test_true = ad.AnnData(X=Y_test)
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

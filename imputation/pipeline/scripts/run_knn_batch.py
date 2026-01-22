import anndata as ad
import numpy as np
import pandas as pd
import os
import time
import json
import time
import tracemalloc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from utils import load_data

tracemalloc.start()

mod1_path = snakemake.input.mod1
mod2_path = snakemake.input.mod2
out_path = snakemake.output.h5ad
log_path = snakemake.output.log

mod1_hvf = pd.read_csv(snakemake.input.mod1_hvf, header=None)[0].tolist()
mod2_hvf = pd.read_csv(snakemake.input.mod2_hvf, header=None)[0].tolist()

X_input, Y_target, split, _ = load_data(mod1_path, mod2_path, mod1_hvf, mod2_hvf)

X_train, X_test = X_input[split == 'train'], X_input[split == 'test']
Y_train, Y_test = Y_target[split == 'train'], Y_target[split == 'test']

start_time = time.time()

pca = PCA(n_components=50)
X_combined = np.vstack((X_train, X_test))
X_combined_pca = pca.fit_transform(X_combined)

X_train_pca = X_combined_pca[:len(X_train)]
X_test_pca = X_combined_pca[len(X_train):]

knn = KNeighborsRegressor(n_neighbors=15)
knn.fit(X_train_pca, Y_train)

Y_test_pred = knn.predict(X_test_pca)

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

import argparse
import scipy.sparse
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

print('Method:', snakemake.wildcards.method)
print('Setting:', snakemake.wildcards.setting)

pred_adata = ad.read_h5ad(snakemake.input.pred)
true_adata = ad.read_h5ad(snakemake.input.pred.replace("imputed.h5ad", "true.h5ad"))

true = np.array(true_adata.X.todense()) if scipy.sparse.issparse(true_adata.X) else np.array(true_adata.X)
pred = np.array(pred_adata.X.todense()) if scipy.sparse.issparse(pred_adata.X) else np.array(pred_adata.X)

print('Shapes:')
print('True:', true.shape)
print('Predicted:', pred.shape)

print('Values:')
print('True:', true[0, :10])
print('Predicted:', pred[0, :10])

spearmans = [spearmanr(true[:, i], pred[:, i])[0] for i in range(true.shape[1])]
pearsons = [pearsonr(true[:, i], pred[:, i])[0] for i in range(true.shape[1])]

scaler = StandardScaler()
true_z = scaler.fit_transform(true)
pred_z = scaler.fit_transform(pred)

pearsons_z = [pearsonr(true[:, i], pred[:, i])[0] for i in range(true.shape[1])]


df = pd.DataFrame({
    "pearson_mean": [np.mean(pearsons)],
    "spearman_mean": [np.mean(spearmans)],
    "pearson_z_mean": [np.mean(pearsons_z)],
})
df.to_csv(snakemake.output[0], sep="\t", index=False)

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import scipy.sparse
import psutil


# def find_indexes(list1, list2):
#     indexes = []
#     for item in list2:
#         index = list1.index(item)
#         indexes.append(index)
#     return indexes

# def generate_gaussian_vectors(lst, length, mean=0, std_dev=1):
#     gaussian_vectors = []
#     for _ in lst:
#         vector = np.random.normal(mean, std_dev, size=(length,))
#         gaussian_vectors.append(vector)
#     return np.array(gaussian_vectors)

def transform_data(adata, hvf=None):
    if hvf is not None:
        hvf = list(hvf)
        adata = adata[:, hvf].copy()
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    X = adata.X.A if isinstance(adata.X, np.matrix) else np.array(adata.X)
    return X

def load_data(input_path, target_path, mod1_hvf=None, mod2_hvf=None, split_key='split', cell_type_key='cell_type'):
    X_mod1 = sc.read_h5ad(input_path)
    X_mod2 = sc.read_h5ad(target_path)

    split = X_mod1.obs[split_key]
    cell_types = X_mod1.obs[cell_type_key]
    
    X_mod1 = transform_data(X_mod1, hvf=mod1_hvf)
    X_mod2 = transform_data(X_mod2, hvf=mod2_hvf)
    
    # input_adata = sc.read_h5ad(input_path)
    # target_adata = sc.read_h5ad(target_path)

    # if mod1_hvf is not None:
    #     input_adata = input_adata[:, mod1_hvf].copy()
    # if mod2_hvf is not None:
    #     target_adata = target_adata[:, mod2_hvf].copy()

    # # move to dense if sparse
    # if scipy.sparse.issparse(input_adata.X):
    #     input_adata.X = input_adata.X.todense()

    # if scipy.sparse.issparse(target_adata.X):
    #     target_adata.X = target_adata.X.todense()

    # X_input = input_adata.X.A if isinstance(input_adata.X, np.matrix) else np.array(input_adata.X)
    # Y_target = target_adata.X.A if isinstance(target_adata.X, np.matrix) else np.array(target_adata.X)
    
    return X_mod1, X_mod2, split, cell_types


def monitor_memory(process, interval, peak_dict):
    """Background thread to monitor peak RSS."""
    peak = 0
    while peak_dict["running"]:
        rss = process.memory_info().rss
        if rss > peak:
            peak = rss
        time.sleep(interval)
    peak_dict["peak_rss"] = peak
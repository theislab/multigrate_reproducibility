import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os
import time
import json
import time
import tracemalloc

import sys
from pathlib import Path
import muon as mu
import numpy as np
import torch
from scipy.sparse import csr_matrix
import networkx as nx
import itertools
import scglue


monae_dir = ("/lustre/groups/ml01/code/anastasia.litinetskaya/monae")

sys.path.append(str(monae_dir))

import src.emb_init as si
from src.train import covel_train
from src.config import configure_dataset

mod1_path = snakemake.input.mod1
mod2_path = snakemake.input.mod2
out_path = snakemake.output.h5ad
log_path = snakemake.output.log

mod1_hvf = pd.read_csv(snakemake.input.mod1_hvf, header=None)[0].tolist()
mod2_hvf = pd.read_csv(snakemake.input.mod2_hvf, header=None)[0].tolist()

try:
    graph_path = mod1_path.replace("rna_train", "guidance-hvf").replace(".h5ad", ".graphml.gz")
    graph = nx.read_graphml(graph_path)
except:
    graph_path = mod1_path.replace("adata_gene", "guidance-hvf").replace(".h5ad", ".graphml.gz")
    graph = nx.read_graphml(graph_path)

print('Loaded graph from', graph_path)

tracemalloc.start()

# data loading and pp
mod1_data = sc.read(mod1_path)
mod2_data = sc.read(mod2_path)

# mod1_data.X = mod1_data.layers['counts'].copy()
# mod2_data.X = mod2_data.layers['counts'].copy()

del mod1_data.layers
del mod2_data.layers

workdir = out_path.replace("imputed.h5ad", "")
os.makedirs(workdir, exist_ok=True)
si.settings.set_workdir(workdir)

elapsed = 0

if 'adt' in mod2_path or 'protein' in mod2_path:
    print('Processing CITE-seq data...')

    # for cite all hvg and all proteins are present in the graph nodes due to how we constructed the graph
    # so no need to subset
    mod1_data = mod1_data[:, mod1_hvf].copy()
    mod2_data = mod2_data[:, mod2_hvf].copy()

    mod1_data.var_names = mod1_data.var_names + '_gex'
    mod2_data.var_names = mod2_data.var_names + '_adt'

    # sc.pp.normalize_total(mod1_data)
    # sc.pp.log1p(mod1_data)
    sc.pp.scale(mod1_data)
    # all of the nodes are hvgs so don't need this any more
    # hvgs = [gene for gene in list(graph.nodes()) if gene.endswith('_gex')]
    # mod1_data = mod1_data[:, hvgs].copy()

    # mu.prot.pp.clr(mod2_data)
    sc.pp.scale(mod2_data)

    mod1_data.obs['cell type'] = mod1_data.obs['cell_type']
    mod2_data.obs['cell type'] = mod2_data.obs['cell_type']

    mod1_data.var['highly_variable'] = True
    mod2_data.var['highly_variable'] = True

    adatas=[mod1_data[mod1_data.obs['split'] == 'train'], mod2_data[mod2_data.obs['split'] == 'train']]
    modal_names=["RNA", "ADT"]
    prob=['Normal', 'Normal']
    rep = [None, None]

    save_path = f"{workdir}/ckpt"
    vertices = sorted(graph.nodes)

    for idx, adata in enumerate(adatas):
        configure_dataset( 
            adata,
            prob[idx], 
            use_highly_variable=True,
            use_rep=rep[idx],
        )

    data = dict(zip(modal_names, adatas))

    start_time = time.time()

    covel = covel_train(
        adatas, 
        graph,
        fit_kws={"directory": save_path},
        config = [modal_names, prob, rep],
        result_path = workdir,
    )   

    end_time = time.time()

    print('CITE done training.')

    # inference
    adatas = [mod1_data[mod1_data.obs['split'] == 'test']]
    modal_names=["RNA"]
    for idx, adata in enumerate(adatas):
        configure_dataset(
            adata,
            prob[idx], 
            use_highly_variable=True,
            use_rep=rep[idx],
        )

    data = dict(zip(modal_names, adatas))
    imputation_X = covel.decode_data("RNA", "ADT", data["RNA"], graph)
    imputation_X = imputation_X.astype(np.float32)
    imputation = sc.AnnData(imputation_X)
    adata_test_pred = imputation

    adata_test_true = mod2_data[mod2_data.obs['split'] == 'test'].copy()
    
elif 'atac' in mod2_path:
    print('Processing multiome data...')
    # pp
    hvgs = [feature for feature in list(graph.nodes()) if not feature.startswith('chr')]
    hvps = [feature for feature in list(graph.nodes()) if feature.startswith('chr')]

    print('Total HVGs in graph:', len(hvgs))
    print('Total HVPs in graph:', len(hvps))

    mod1_hvf = list(set(hvgs).intersection(set(mod1_hvf)))
    mod2_hvf = list(set(hvps).intersection(set(mod2_hvf)))

    print('Mod1 HVGs in data and graph:', len(mod1_hvf))
    print('Mod2 HVPs in data and graph:', len(mod2_hvf))

    mod1_data = mod1_data[:, mod1_hvf].copy()
    mod2_data = mod2_data[:, mod2_hvf].copy()

    print('After subsetting data')
    print('Mod1 data shape:', mod1_data.shape)
    print('Mod2 data shape:', mod2_data.shape)

    # sc.pp.normalize_total(mod1_data)
    # sc.pp.log1p(mod1_data)
    sc.pp.scale(mod1_data)
    # mod1_data = mod1_data[:, hvgs].copy()
    
    # sc.pp.normalize_total(mod2_data)
    # sc.pp.log1p(mod2_data)
    sc.pp.scale(mod2_data)
    # mod2_data = mod2_data[:, hvps].copy()

    mod1_data.obs['cell type'] = mod1_data.obs['cell_type']
    mod2_data.obs['cell type'] = mod2_data.obs['cell_type']

    mod1_data.var['highly_variable'] = True
    mod2_data.var['highly_variable'] = True

    adatas=[mod1_data[mod1_data.obs['split'] == 'train'], mod2_data[mod2_data.obs['split'] == 'train']]
    modal_names=["RNA", "ATAC"]
    prob=['Normal', 'Normal']
    rep = [None, None]

    save_path = f"{workdir}/ckpt"
    vertices = sorted(graph.nodes)

    for idx, adata in enumerate(adatas):
        configure_dataset( 
            adata,
            prob[idx], 
            use_highly_variable=True,
            use_rep=rep[idx],
        )

    data = dict(zip(modal_names, adatas))

    start_time = time.time()

    covel = covel_train(
        adatas, 
        graph,
        fit_kws={"directory": save_path},
        config = [modal_names, prob, rep],
        result_path = workdir,
    )   

    end_time = time.time()

    print('Multiome done training.')

    # inference
    adatas = [mod1_data[mod1_data.obs['split'] == 'test']]
    modal_names=["RNA"]
    for idx, adata in enumerate(adatas):
        configure_dataset(
            adata,
            prob[idx], 
            use_highly_variable=True,
            use_rep=rep[idx],
        )

    data = dict(zip(modal_names, adatas))
    imputation_X = covel.decode_data("RNA", "ATAC", data["RNA"], graph)
    imputation_X = imputation_X.astype(np.float32)
    imputation = sc.AnnData(imputation_X)
    adata_test_pred = imputation

    adata_test_true = mod2_data[mod2_data.obs['split'] == 'test'].copy()

adata_test_pred.write_h5ad(out_path)
adata_test_true.write_h5ad(out_path.replace("imputed.h5ad", "true.h5ad"))

elapsed += end_time - start_time

_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

runtime_data = {
    "method": os.path.basename(__file__),
    "time_sec": round(elapsed, 2),
    "peak_memory_mb": round(peak / (1024 ** 2), 2),
}

with open(log_path, "w") as f:
    json.dump(runtime_data, f, indent=2)


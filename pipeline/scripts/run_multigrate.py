import time
start_time = time.time()

import scanpy as sc
import pandas as pd
import multimil as mtm
from pathlib import Path
import shutil

from matplotlib import pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np

import torch
import scvi
from utils import get_existing_checkpoints
import scipy
import sklearn
import scib

import warnings
warnings.filterwarnings('ignore')

import os
import wandb

print('--- %s seconds ---' % (time.time()-start_time))

def run_multigrate(adata1, adata2, split_key, params, hash, task, metrics=False, batch_key=None, label_key=None, **kwargs):

    print('============ Multigrate training ============')

    torch.set_float32_matmul_precision('medium')

    json_config = {}

    method = 'multigrate_test_in_train'

    json_config['method'] = method
    json_config['params'] = params
    json_config['my_hash'] = str(hash)
    json_config['task'] = task
    json_config['adata1'] = str(adata1)
    json_config['adata2'] = str(adata2)

    print('wandb init...')
    wandb.init(project="neurips-test-in-train", entity='recomb', name=task + '_' + hash, id=task + '_' + hash, config=json_config)

    setup_params = {
        "rna_indices_end": params['rna_indices_end'],
        "categorical_covariate_keys": [batch_key],
    }

    model_params = {
        "z_dim": params['z_dim'],
        "cond_dim": params['cond_dim'],
        "mix": params['mix'],
        "integrate_on": params['integrate_on'],
        "modality_alignment": params['modality_alignment'],
        "alignment_type": params['alignment_type'],
    }
    loss_params = {
        'kl': params['kl'],
        'integ': params['integ'],
        '0': params['mod_0'],
        '1': params['mod_1'],
    }
    umap_colors = params['umap_colors'].strip('][').replace('\'', '').replace('\"', '').split(', ')
    train_params = {
        "max_epochs": params['train_max_epochs'],
    }

    # train params
    lr = params['lr']
    batch_size = params['batch_size']
    seed = params['seed']

    scvi.settings.seed = seed

    dfs = []

    # split into train and test
    test_idx = adata2.obs[split_key] == 'test'

    adata2_test = adata2[test_idx].copy()
    adata2_train = adata2[~test_idx].copy()

    adata1_test = adata1[test_idx].copy()
    adata1_train = adata1[~test_idx].copy()

    # free memory
    del adata1
    del adata2

    ########################
    ######## TRAIN #########
    ########################

    losses = ['nb', 'mse']
    print('Organizing multiome anndatas...')
    adata = mtm.data.organize_multimodal_anndatas(
        adatas=[[adata1_train, adata1_test], [adata2_train, None]],
        )
    
    print('Setting up anndata...')
    mtm.model.MultiVAE.setup_anndata(
        adata, 
        **setup_params
    )

    print('Initializing the model...')
    vae = mtm.model.MultiVAE(
        adata,
        losses=losses,
        loss_coefs=loss_params,
        **model_params,
    )

    os.makedirs(f'data/{method}/{task}/{hash}/', exist_ok=True)
    print('Starting training...')

    torch.cuda.reset_peak_memory_stats()
    try:
        start_train_time = time.time()
        vae.train(
            lr=lr,
            batch_size=batch_size, 
            **train_params
        )
        train_time = time.time()-start_train_time
    except ValueError:
        print("nan's in training, aborting...")
        df = {}
        df['method'] = method
        df = pd.DataFrame.from_dict(df, orient='index').T
        dfs.append(df)
        df = pd.concat(dfs)
        return df

    peak_mem_in_gb = torch.cuda.max_memory_allocated() / (1024**3)

    train_metrics = {
        'train_time': train_time,
        'peak_mem_in_gb': peak_mem_in_gb,
    }
    train_metrics = pd.DataFrame(train_metrics, index=[0])
    train_metrics.to_csv(f'data/{method}/{task}/{hash}/train_metrics.csv')

    print('Starting inference...')
    vae.get_model_output(batch_size=batch_size)

    print('Calculating neighbors...')
    sc.pp.neighbors(adata, use_rep="X_multiMIL")
    sc.tl.umap(adata)

    sc.pl.umap(
        adata,
        color=umap_colors,
        ncols=1,
        show=False,
    )

    plt.savefig(f'data/{method}/{task}/{hash}/train_umap.png', bbox_inches="tight")
    plt.close()

    vae.plot_losses(save=f'data/{method}/{task}/{hash}/train_losses.png')

    #########################################
    ###### PREDICT WITHOUT FINETUNING #######
    #########################################

    wandb.log({
        "train_time": train_time,
        "peak_mem_in_gb": peak_mem_in_gb,
    })

    print('Starting imputation...')
    vae.impute()

    prediction_metrics = {'train': [], 'test': []}
    for split, adata_tmp in zip(['train', 'test'], [adata2_train, adata2_test]):
        if scipy.sparse.issparse(adata_tmp.X):
            adata_tmp.X = adata_tmp.X.toarray()

        prediction_metrics[split] = {
            'mse': sklearn.metrics.mean_squared_error(adata_tmp.X, adata[adata.obs[split_key] == split].obsm['imputed_modality_1']),
            'r2_score': sklearn.metrics.r2_score(adata_tmp.X, adata[adata.obs[split_key] == split].obsm['imputed_modality_1']),
            'pearson': scipy.stats.pearsonr(adata_tmp.X.ravel(), adata[adata.obs[split_key] == split].obsm['imputed_modality_1'].ravel())[0],
            'spearman': scipy.stats.spearmanr(adata_tmp.X.ravel(), adata[adata.obs[split_key] == split].obsm['imputed_modality_1'].ravel())[0],
        }
       
    df = pd.DataFrame(prediction_metrics)
    
    df.to_csv(f'data/{method}/{task}/{hash}/prediction_metrics.csv')

    df['method'] = method
    epoch = np.max(vae.history['train_loss_step'].index)
    df['epoch'] = epoch
    df['query_epoch'] = 0

    dfs.append(df)

    metrics_df = {}
    if metrics is True:
        print('Calculating metrics...')
        scib.metrics.cluster_optimal_resolution(
            adata,
            label_key=label_key,
            cluster_key='cluster',
            use_rep='X_multiMIL',
        )
        metrics_df['ARI'] = scib.metrics.ari(adata, cluster_key='cluster', label_key=label_key)
        metrics_df['NMI'] = scib.metrics.nmi(adata, cluster_key='cluster', label_key=label_key)
        metrics_df['Isolated label score ASW'] = scib.metrics.isolated_labels_asw(
            adata, 
            label_key = label_key, 
            batch_key = batch_key, 
            embed = 'X_multiMIL',
        )
        metrics_df['Label ASW'] = scib.metrics.silhouette(
            adata,
            label_key = label_key,
            embed = 'X_multiMIL',
        )
        metrics_df['Batch ASW'] = scib.metrics.silhouette_batch(
            adata, 
            batch_key = batch_key,
            label_key = label_key,
            embed = 'X_multiMIL',
        )
        metrics_df['Graph Connectivity'] = scib.metrics.graph_connectivity(
            adata,
            label_key = label_key
        )
        metrics_df['Overall Score'] = 0.6 * np.mean([
            metrics_df['ARI'],
            metrics_df['NMI'],
            metrics_df['Isolated label score ASW'],
            metrics_df['Label ASW']]
            ) + 0.4 * np.mean([metrics_df['Batch ASW'], metrics_df['Graph Connectivity']])
        wandb.log(metrics_df)
        metrics_df = pd.DataFrame.from_dict(metrics_df, orient='index').T
        metrics_df.to_csv(f'data/{method}/{task}/{hash}/scib.csv')

    wandb.log({
        'epoch': epoch,
        'query_epoch': 0,
        'train_mse': prediction_metrics['train']['mse'],
        'test_mse': prediction_metrics['test']['mse'],
        'train_r2_score': prediction_metrics['train']['r2_score'],
        'test_r2_score': prediction_metrics['test']['r2_score'],
        'train_pearson': prediction_metrics['train']['pearson'],
        'test_pearson': prediction_metrics['test']['pearson'],
        'train_spearman': prediction_metrics['train']['spearman'],
        'test_spearman': prediction_metrics['test']['spearman'],
        }
    )

    wandb.finish()

    df = pd.concat(dfs)
    return df

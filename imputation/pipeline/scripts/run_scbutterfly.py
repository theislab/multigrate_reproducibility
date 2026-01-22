import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os
import time
import json
import time
import tracemalloc
from muon import atac as ac

from scipy.sparse import csr_matrix

from scButterfly.butterfly import Butterfly
from scButterfly.data_processing import RNA_data_preprocessing, CLR_transform
from scButterfly.train_model_cite import Model
import torch
import torch.nn as nn

butterfly = Butterfly()

tracemalloc.start()

mod1_path = snakemake.input.mod1
mod2_path = snakemake.input.mod2
out_path = snakemake.output.h5ad
log_path = snakemake.output.log

mod1_hvf = pd.read_csv(snakemake.input.mod1_hvf, header=None)[0].tolist()
mod2_hvf = pd.read_csv(snakemake.input.mod2_hvf, header=None)[0].tolist()

# data loading and pp
mod1_data = sc.read(mod1_path)
mod2_data = sc.read(mod2_path)

mod1_data = mod1_data[:, mod1_hvf].copy()
mod2_data = mod2_data[:, mod2_hvf].copy()
mod1_data.var['highly_variable'] = True
mod2_data.var['highly_variable'] = True

# mod1_data.X = mod1_data.layers['counts'].copy()
# mod2_data.X = mod2_data.layers['counts'].copy()

del mod1_data.layers
del mod2_data.layers

mod1_data.X = csr_matrix(mod1_data.X)
mod2_data.X = csr_matrix(mod2_data.X)

# find numerical train/val/test indices
tmp = mod1_data.copy()
tmp.obs.reset_index(inplace=True)

train_val_id = tmp[tmp.obs['split'] == 'train'].obs_names.tolist()
print('len of train_id:', len(train_val_id))

# shufle and random split train and validation
import numpy as np
np.random.seed(0)
np.random.shuffle(train_val_id)
n_train = int(len(train_val_id) * 0.8)
train_id = train_val_id[:n_train]
validation_id = train_val_id[n_train:]

test_id = tmp[tmp.obs['split'] == 'test'].obs_names.tolist()
print('len of test_id:', len(test_id))

train_id, validation_id, test_id 
train_id_r = train_id.copy()
train_id_a = train_id.copy()
validation_id_r = validation_id.copy()
validation_id_a = validation_id.copy()
test_id_r = test_id.copy()
test_id_a = test_id.copy()

del tmp

output_path = out_path.replace('.h5ad', '')
os.makedirs(output_path, exist_ok=True)

if 'adt' in mod2_path or 'protein' in mod2_path:
    mod1_data = RNA_data_preprocessing(
        mod1_data,
        normalize_total=False,
        log1p=False,
        use_hvg=False,
        n_top_genes=None,
        save_data=False,
        file_path=None,
        logging_path=None
    )
    # mod2_data = CLR_transform(mod2_data)[0]

    RNA_input_dim = len([i for i in mod1_data.var['highly_variable'] if i])
    ADT_input_dim = mod2_data.X.shape[1]

    R_kl_div = 1 / RNA_input_dim * 20
    A_kl_div = 1 / 150
    kl_div = R_kl_div + A_kl_div

    model = Model(
        R_encoder_nlayer = 2, 
        A_encoder_nlayer = 2,
        R_decoder_nlayer = 2, 
        A_decoder_nlayer = 2,
        R_encoder_dim_list = [RNA_input_dim, 256, 128],
        A_encoder_dim_list = [ADT_input_dim, 128, 128],
        R_decoder_dim_list = [128, 256, RNA_input_dim],
        A_decoder_dim_list = [128, 128, ADT_input_dim],
        R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list = [nn.LeakyReLU(), nn.Identity()],
        translator_embed_dim = 128, 
        translator_input_dim_r = 128,
        translator_input_dim_a = 128,
        translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer = 1,
        discriminator_dim_list_R = [128],
        discriminator_dim_list_A = [128],
        discriminator_act_list = [nn.Sigmoid()],
        dropout_rate = 0.1,
        R_noise_rate = 0.5,
        A_noise_rate = 0,
        chrom_list = [],
        logging_path = None,
        RNA_data = mod1_data,
        ATAC_data = mod2_data
    )

    start_time = time.time()
    model.train(
        R_encoder_lr = 0.001,
        A_encoder_lr = 0.001,
        R_decoder_lr = 0.001,
        A_decoder_lr = 0.001,
        R_translator_lr = 0.001,
        A_translator_lr = 0.001,
        translator_lr = 0.001,
        discriminator_lr = 0.005,
        R2R_pretrain_epoch = 100, 
        A2A_pretrain_epoch = 100, 
        lock_encoder_and_decoder = False,
        translator_epoch = 200,
        patience = 50,
        batch_size = 64,
        r_loss = nn.MSELoss(size_average=True),
        a_loss = nn.MSELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        loss_weight = [1, 2, 1, R_kl_div, A_kl_div, kl_div],
        train_id_r = train_id_r,
        train_id_a = train_id_a,
        validation_id_r = validation_id_r, 
        validation_id_a = validation_id_a, 
        output_path = output_path,
        seed = 19193,
        kl_mean = True,
        R_pretrain_kl_warmup = 50,
        A_pretrain_kl_warmup = 50,
        translation_kl_warmup = 50,
        load_model = None,
        logging_path = None
    )
    end_time = time.time()

    _, R2A_predict = model.test(
        test_id_r = test_id_r,
        test_id_a = test_id_a, 
        model_path = None,
        load_model = False,
        output_path = None,
        test_cluster = False,
        test_figure = False,
        output_data = False,
        return_predict = True
    )

elif 'atac' in mod2_path:
    mod2_data = mod2_data[ :, mod2_data.var_names.str.startswith('chr')]
    mod2_data.var['chrom'] = mod2_data.var_names.str.split('-').str[0].to_list()

    # ac.pp.binarize(mod2_data)
    # ac.pp.tfidf(mod2_data, scale_factor=1e4)
    
    # sc.pp.highly_variable_genes(mod2_data, n_top_genes=20000, subset=True)

    butterfly.load_data(mod1_data, mod2_data, train_id, test_id, validation_id)

    butterfly.data_preprocessing(
        binary_data=False,
        filter_features=False,
        tfidf=False,
        normalize=True,
        use_hvg=False,
        normalize_total=False,
        log1p=False,
        n_top_genes=None,
        )

    # count peaks in each chromosome
    chrom_list = []
    last_one = ''
    for i in range(len(butterfly.ATAC_data_p.var.chrom)):
        temp = butterfly.ATAC_data_p.var.chrom[i]
        if temp[0 : 3] == 'chr':
            if not temp == last_one:
                chrom_list.append(1)
                last_one = temp
            else:
                chrom_list[-1] += 1
        else:
            chrom_list[-1] += 1

    butterfly.augmentation(aug_type=None)

    butterfly.construct_model(chrom_list=chrom_list)

    start_time = time.time()
    butterfly.train_model(
        R2R_pretrain_epoch = 100,
        A2A_pretrain_epoch = 100,
        translator_epoch = 200,
        output_path = output_path,
    )
    end_time = time.time()

    _, R2A_predict = butterfly.test_model()
else:
    raise ValueError("mod2_path does not contain either 'adt' or 'atac'")

adata_test_pred = R2A_predict
adata_test_pred.write_h5ad(out_path)

adata_test_true = mod2_data[test_id_a, :].copy()
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

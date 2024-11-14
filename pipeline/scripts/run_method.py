import time 
start_time = time.time()
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import scanpy as sc
import ast

from run_multigrate import run_multigrate
from run_multigrate_no_rna import run_multigrate_no_rna

print('--- %s seconds ---' % (time.time() - start_time))

METHOD_MAP = dict(
    multigrate=dict(function=run_multigrate),
    multigrate_no_rna=dict(function=run_multigrate_no_rna),
)

params = snakemake.params.params

method_params = ast.literal_eval(params['params'].replace('nan', 'None')) # this is dict
input1 = params['input1']
input2 = params['input2']
split = params['split']
metrics = params['metrics']
batch_key = params['batch_key']
label_key = params['label_key']
h = params['hash']
method = params['method']
task = params['task']
output_file = snakemake.output.tsv

method_function = METHOD_MAP[method]['function']

adata1 = sc.read_h5ad(input1)
adata2 = sc.read_h5ad(input2)
df = method_function(
    adata1=adata1,
    adata2=adata2,
    split_key=split,
    output_file=output_file,
    params=method_params,
    hash=h,
    method=method,
    task=task,
    metrics=metrics,
    batch_key=batch_key,
    label_key=label_key,
)

df['hash'] = h
df['method_params'] = params['params']
df['task'] = task
df['method'] = method
df.to_csv(output_file, sep='\t')

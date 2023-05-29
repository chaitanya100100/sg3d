import matplotlib.pyplot as plt
from collections import defaultdict

import pandas as pd
import numpy as np


fpath = "/vision/u/chpatel/STTran_3d/test_cp.md"
with open(fpath, 'r') as fpt:
    lines = fpt.readlines()

metric_types_all = ['R@10','R@20','R@50','R@100']
all_metrics = {
    'with constraint': {},
    'semi constraint': {},
    'no constraint': {},
}

exp_name = None
epoch = None
dct = None
sg_mode = None
metric_type, metric_value = None, None
constraint_type = None

def finishup():
    global exp_name, epoch, dct, sg_mode, metric_type, metric_value, constraint_type
    if exp_name is None:
        return
    
    for cty, dd in dct.items():
        if exp_name not in all_metrics[cty]:
            all_metrics[cty][exp_name] = {}
        all_metrics[cty][exp_name] = dd


    exp_name = None
    epoch = None
    dct = None
    sg_mode = None
    metric_type, metric_value = None, None
    constraint_type = None    

for ln in lines:
    ln = ln.strip()
    if not ln:
        continue
    if ln.startswith('##'):
        finishup()
        exp_name = ln[2:]
        dct = {}
        continue
    if ln.startswith('save the checkpoint'):
        epoch = ln.replace('save the checkpoint after ', '').replace(' epochs', '')
        epoch = int(epoch)
        continue
    if ln.startswith('==='):
        sg_mode = ln.replace('=', '')
        continue
    if ln.startswith('R@'):
        metric_type, metric_value = ln.split(' ')
        metric_type = metric_type.replace(':', '')
        if epoch is None: epoch = 9
        if constraint_type is None: constraint_type = 'with constraint'

        if constraint_type not in dct: dct[constraint_type] = {}
        if epoch not in dct[constraint_type]: dct[constraint_type][epoch] = {}
        dct[constraint_type][epoch][metric_type] = float(metric_value)
    if ln.startswith('Epoch '): # lr reduction
        continue
    if ln.startswith('----'):
        constraint_type = ln.replace('-', '')
        continue

finishup()

def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    index = pd.MultiIndex.from_tuples(df.index)
    df = pd.Series(df.values.reshape(-1), index=index)
    # df = df.unstack(level=-1)
    # df.columns = df.columns.map("{0[1]}".format)
    return df

new_metrics = {'exp name': []}
new_metrics.update({m: [] for m in metric_types_all})
for exp_name, res in all_metrics['with constraint'].items():
    
    epochs = list(res.keys())
    max_epoch_idx = np.argmax([v['R@50'] for k, v in res.items()])
    max_epoch = epochs[max_epoch_idx]
    
    new_metrics['exp name'].append(exp_name)
    for m in metric_types_all:
        new_metrics[m].append(res[max_epoch][m])


df = pd.DataFrame(new_metrics, columns=['exp name',]+metric_types_all)
df = df.sort_values(df.columns[0], ascending = True)
print(df)
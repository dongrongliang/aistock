# -*- coding: utf-8 -*-
# !/usr/bin/env python
import numpy as np
import pandas as pd

def astype_map(data, type_map):
    for k, v in type_map.items():
        data[k] = data[k].astype(v)

def astype_list(data, type_list):
    type_map = {}
    for column, v in zip(data.columns.values, type_list):
        type_map[column] = v
    astype_map(data, type_map)

def save(data, file_name='data'):
    np.savez(file_name, data=data.values, index=data.index, columns=data.columns, dtype=data.dtypes)


def load(file_name='data'):
    data = np.load(file_name if file_name[-4:] == '.npz' else file_name + '.npz', allow_pickle=True)
    ret = pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])
    astype_list(ret, data['dtype'])
    return ret
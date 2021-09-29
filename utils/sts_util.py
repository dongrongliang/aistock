# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd
import numpy as np
import math
from scipy import stats
import scipy
from sklearn import preprocessing
from scipy.ndimage.interpolation import shift
import warnings
from itertools import groupby

warnings.filterwarnings("ignore")


def array_corr(hist_1, hist_2, corr_type=3):
    h_1_s = pd.Series(hist_1)
    h_2_s = pd.Series(hist_2)
    if corr_type == 1:
        p_corr = h_1_s.corr(h_2_s, 'spearman')
    elif corr_type == 2:
        p_corr = h_1_s.corr(h_2_s, 'kendall')
    else:
        p_corr = h_1_s.corr(h_2_s, 'pearson')
    return p_corr


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def group_freq(group):
    return {'count': group.value_counts(normalize=True)}


## ma
def cal_ma(data_array, n=5, central_window=False):
    data_array = np.array(data_array)
    N = len(data_array)
    ma_array = np.zeros(N - n + 1)

    if central_window:
        kernel = np.ones((n - 1) * 2 + 1)
        ma_array = np.convolve(data_array, kernel, mode='same')
        ma_array[n - 1:-n + 1] = ma_array[n - 1:-n + 1] / ((n - 1) * 2 + 1)
        for i in range(n - 1):
            ma_array[i] = ma_array[i] / (n + i)
            ma_array[-i - 1] = ma_array[-i - 1] / (n + i)
        return ma_array
    else:
        for i in range(n):
            ma_array = ma_array + data_array[i:i - n + 1] if i < n - 1 else ma_array + data_array[i:]
        ma_array = ma_array / n
        return np.concatenate([np.array([0] * (n - 1)), ma_array])


## el
def cal_el(data_array, n=5, level_max=True):
    data_array = np.array(data_array)
    N = len(data_array)

    for i in range(n):
        pass_array = data_array[i:i - n + 1] if i < n - 1 else data_array[i:]
        el_array = np.row_stack([el_array, pass_array]) if i > 0 else pass_array
    el_array = el_array.max(axis=0) if level_max else el_array.min(axis=0)
    return np.concatenate([np.array([0] * (n - 1)), el_array])


# moving variance
def cal_mv(data_array, n=5, norm=True):
    data_array = np.array(data_array)
    N = len(data_array)

    for i in range(n):
        pass_array = data_array[i:i - n + 1] if i < n - 1 else data_array[i:]
        stack_array = np.row_stack([stack_array, pass_array]) if i > 0 else pass_array
    if norm:
        min_array = stack_array.min(axis=0)
        max_array = stack_array.max(axis=0)
        stack_array = (stack_array - min_array) / (max_array - min_array)
    mean_array = stack_array.mean(axis=0)
    mv_array = np.sqrt(np.square(stack_array - mean_array).sum(0) / n)

    return np.concatenate([np.array([0] * (n - 1)), mv_array])


def fill_duplicate_adjacency(lst, fill=None, keep_last=True):
    new_L = []
    for x, group in groupby(lst):
        new_group = list(group)
        if len(new_group) > 1:
            if keep_last:
                new_group[:-1] = [fill] * len(new_group[:-1])
            else:
                new_group[1:] = [fill] * len(new_group[1:])
        new_L.extend(new_group)
    return new_L


def intersection_points(a, b, real_cross=True,
                        strict=False, keep_adjacency_point=False, fuzzy_window=2):
    assert fuzzy_window >=1
    a = np.array(a)
    b = np.array(b)
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError('The arrays must be single dimensional and the same length')

    greater = a > b
    less = a < b
    equal = a == b

    # shift will recover the loss precision, eg:shift([1,2.33],-1) = [2.330003,0]
    right_greater = np.concatenate([greater[1:], [False]])
    right_less = np.concatenate([less[1:], [False]])
    left_greater = np.concatenate([[False], greater[:-1]])
    left_less = np.concatenate([[False], less[:-1]])
    if real_cross:
        for window in range(fuzzy_window - 1):
            window += 2
            near_equal = np.concatenate([[False] * (window - 1), equal[:-window + 1]])
            left_greater = (left_greater) | ((np.concatenate([[False] * window, greater[:-window]])) & (near_equal))
            left_less = (left_less) | ((np.concatenate([[False] * window, less[:-window]])) & (near_equal))

    if real_cross:
        upward_cross = (left_less) & (right_greater)
        downward_cross = (left_greater) & (right_less)
    else:
        upward_cross = (left_less) & ~(right_less)
        downward_cross = (left_greater) & ~(right_greater)

    # upward_cross = (left_less) & (right_greater)
    # downward_cross = (left_greater) & (right_less)
    if strict:
        equal_cross = a == b
        upward_cross = (upward_cross) & (equal_cross)
        downward_cross = (downward_cross) & (equal_cross)
    # start points and end points may reach the equal line or reach the bifurcation point
    upward_cross[:1] = False
    downward_cross[:1] = False
    # upward_cross[-1:] = False
    # downward_cross[-1:] = False

    #     upward_cross[0] = (a[1] > b[1]) & (a[0] == b[0])
    #     downward_cross[0] = (a[1] < b[1]) & (a[0] == b[0])
    #     upward_cross[-1] = (a[-2] < b[-2]) & (a[-1] == b[-1])
    #     downward_cross[-1] = (a[-2] > b[-2]) & (a[-1] == b[-1])

    if ~keep_adjacency_point:
        upward_cross = fill_duplicate_adjacency(upward_cross, fill=False)
        downward_cross = fill_duplicate_adjacency(downward_cross, fill=False)

    return upward_cross, downward_cross


def fuzzy_equal(a, b, window_size=2, cal_future=True):
    a = np.array(a)
    b = np.array(b)

    equal_array = a == b

    for w in range(window_size):
        _w = w + 1
        if cal_future:
            equal_array = (equal_array) | (np.concatenate([equal_array[_w:], [False] * _w]))
        equal_array = (equal_array) | (np.concatenate([[False] * _w, equal_array[:-_w]]))
    return equal_array

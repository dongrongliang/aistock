# -*- coding: utf-8 -*-
# !/usr/bin/env python

import psutil
import pynvml

def get_memory_utilization():
    info = psutil.virtual_memory()
    return info.percent

def get_free_memory():
    info = psutil.virtual_memory()
    return round((float(info.free) / 1024 / 1024 / 1024), 2)

def get_loaded_img_treshold_dist():
    gb_per_img = 149.85 / 113.0
    treshold = int(get_free_memory()*0.95 / gb_per_img)
    return treshold

def get_loaded_img_treshold():
    gb_per_img = 47 / 59
    treshold = int(get_free_memory()*0.95 / gb_per_img)
    return treshold

def get_gpu_memory(gpu_device):
    # 获取当前gpu使用显存
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_rate = meminfo.free * 1.0 / meminfo.total

    return used_rate
# -*- coding: utf-8 -*-
# !/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader
import os
from utils.np_utils import *
from utils.device_utils import *
import random
import gc
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class DataSetXY(Dataset):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, period=4 * 28, neutral_sampling_prob=1, multi_label=False,
                 drop_y=False, y_radius=6, y_recent_fill=10):
        super().__init__()
        self.drop_y = drop_y
        self.y_radius = y_radius
        self.y_recent_fill = y_recent_fill
        self.multi_label = multi_label
        self.neutral_sampling_prob = neutral_sampling_prob
        self.stock_num = len(os.listdir(root))
        self.data_lst = []
        self.positive_count = 0
        self.negative_count = 0
        self.load_data2memory(root, period)
        self.neutral_count = self.__len__() - self.positive_count - self.negative_count
        self.sample_prob = [self.neutral_count / self.__len__(),
                            self.positive_count / self.__len__(),
                            self.negative_count / self.__len__()
                            ]

        self.reverse_sample_prob = [1 - self.neutral_count / self.__len__(),
                                    1 - self.positive_count / self.__len__(),
                                    1 - self.negative_count / self.__len__()
                                    ]
        print('Sample prob {neutral,positive,negative}:%s' % str(self.sample_prob))

    def load_data2memory(self, root, period):

        for j, stoch_fn in enumerate(os.listdir(root)):
            data_path = os.path.join(root, stoch_fn)
            data = load(data_path)
            data = data.sort_values('date').drop('date', axis=1).reset_index(drop=True).astype(float)

            for i, row in data.iterrows():
                y = int(row.y)
                if self.multi_label:
                    multil_y = np.array(data.loc[i - period + 1:i, 'y'].tolist())
                if i + 1 > period:

                    if self.drop_y:
                        samples = data.loc[i - period + 1:i].drop('y', axis=1).to_numpy().T
                    else:
                        samples = data.loc[i - period + 1:i]
                        samples.loc[i - self.y_radius:i] = self.y_recent_fill
                        samples = samples.to_numpy().T

                    if y == 0:
                        if random.random() <= self.neutral_sampling_prob:
                            if self.multi_label:
                                self.data_lst.append((samples, multil_y))
                            else:
                                self.data_lst.append((samples, y))
                    else:
                        if self.multi_label:
                            self.data_lst.append((samples, multil_y))
                        else:
                            self.data_lst.append((samples, y))
                        if y == 1:
                            self.positive_count += 1
                        elif y == 2:
                            self.negative_count += 1
            del(data)
            gc.collect()

            # 计算内存
            progress_step = int((j + 1) * 10 / len(os.listdir(root)))
            now_memory = get_memory_utilization()
            print('[' + '>' * progress_step + ' ' * (10 - progress_step) + ']' + '%s/%s' % (
                j + 1, len(os.listdir(root))) +
                  ', Used memory %s%%' % now_memory + '\r', end='')

            assert now_memory < 95, ('内存超载，当前加载股票数:%s, 当前样本数:%s' % (str(j+1), str(len(os.listdir(root)))))

        self.data_lst = tuple(self.data_lst)

    def __getitem__(self, index):

        data, target = self.data_lst[index]

        x = torch.from_numpy(data).type(torch.FloatTensor)
        if self.multi_label:
            target = torch.from_numpy(target)
        # target = torch.LongTensor(int(target))

        return x, target

    def __len__(self):
        return len(self.data_lst)

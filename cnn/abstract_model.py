# -*- coding: utf-8 -*-
# !/usr/bin/env python

import datetime

import numpy as np
import torch
import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import time
from data_source.data_loader import DataSetXY, DataLoader
import math
from cnn.matrix import *


class BaseModel(object):

    def __init__(self, model_name, mode='train'):

        self.mode = mode
        self.model_name = model_name

        self.model = None
        self.loss = None
        self.optimizer = None

        self.multi_label = False

    def set_hyper_paras(self, p):

        self.model_id = p.get('model_id', '0')

        self.weight_path = os.path.join(PROJECT_PATH, "weights/model_%s.pth" % self.model_id)
        print(self.weight_path)

        self.loss_type = p.get('loss_type', 'ce')

        self.eval_model = p.get('eval_model', True)

        self.period = p.get('period', 4 * 28)

        self.schedule_learn = p.get('schedule_learn', False)

        self.features_num = p.get('features_num', 16)

        self.num_class = p.get('num_class', 3)

        self.optimizer = p.get('optimizer', 'adam')
        self.weight_decay = p.get('weight_decay', 0.0001)
        self.learning_rate = p.get("learning_rate", 0.05)

        self.drop_y = p.get("drop_y", True)
        if not self.drop_y:
            self.features_num += 1
        self.y_radius = p.get("y_radius", 6)
        self.y_recent_fill = p.get("y_recent_fill", 10)

    def create_model(self):
        if self.mode == 'test':
            # 以预测为目的的模型初始化，初始化self.generator
            self.generator = object
            pass
        else:
            # 以训练为目的的模型初始化
            self.generator = object
            pass

    def init_model(self):
        return None

    def init_loss(self):
        self.loss = None

        print('Initialize loss...')

    def init_dataset(self, train_path, period=4 * 20, neutral_sampling_prob=0.5):

        return DataSetXY(train_path, period, neutral_sampling_prob)

    def train(self, train_path, save_model_id, epochs, neutral_sampling_prob=1,
              test_path=None, loss_tresh=0.01, EarlyStopping=True, patience=1,
              loadInMemory=False, draw_history=False):
        return

    def predict(self, x):

        y_pred = self.model.predict(x, batch_size=1)[0]

        return y_pred

    def predict_trend(self, data):

        if self.model is None:
            print('Model need to be created first')
            return None
        assert type(data) == np.ndarray

        y_pred = self.predict(data)

        return y_pred

    def evaluate(self, val_dataset, batch_size):
        bsize = 1
        for i in range(batch_size):
            b = batch_size - i
            if val_dataset.__len__() % b == 0:
                bsize = b
                break
        samples_per_epoch = val_dataset.__len__()
        steps_per_epoch = int(math.ceil(samples_per_epoch / bsize))
        print('Eval batch size:%s' % bsize)
        print('Eval samples_per_epoch:%s' % samples_per_epoch)
        print('Eval steps_per_epoch:%s' % steps_per_epoch)

        data_loader = DataLoader(
            val_dataset,
            batch_size=bsize,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )

        if self.loss is None:
            self.init_loss(val_dataset.reverse_sample_prob)
        if self.eval_model:
            self.model.eval()
        with torch.no_grad():
            for i, (x, label) in enumerate(data_loader):
                x, label = x.cuda(), label.cuda()
                # compute output
                pred = self.model(x)

                loss = self.loss(pred, label)

                # accumulate outputs
                outputs = torch.cat([outputs, pred]) if i > 0 else pred
                targets = torch.cat([targets, label]) if i > 0 else label
                losses = torch.cat([losses, loss.unsqueeze(dim=0)], dim=0) if i > 0 else loss.unsqueeze(dim=0)

                print("Eval Step: %d/%d" % (i, steps_per_epoch) + '\r', end='')

            mean_loss = round(losses.mean().data.item(), 6)
            correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = stock_accuracy(outputs, targets,
                                                                                                multi_label=self.multi_label)
            correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = correct_prob.data.item(), \
                                                                                 var_correct_prob.data.item(), \
                                                                                 netraul_mistake_prob.data.item(), \
                                                                                 reverse_prob.data.item()
            correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = round(correct_prob, 3), \
                                                                                 round(var_correct_prob, 3), \
                                                                                 round(netraul_mistake_prob, 3), \
                                                                                 round(reverse_prob, 3)

        return mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob

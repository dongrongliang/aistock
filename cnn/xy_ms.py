from cnn.abstract_model import *
from cnn.hrnet import StockHRNet
import torch
import torch.nn as nn
from data_source.data_loader import DataSetXY, DataLoader
import math
from torch.optim import lr_scheduler
import shutil
from tensorboardX import SummaryWriter
import functools
import gc
from cnn.matrix import *
from cnn.loss import *
from cnn.base_net import *
from cnn.half_res import *
from cnn.trm import stockBERT
from utils.np_utils import load, save
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from cnn.matrix import stock_accuracy_np
from trend_ana.sts_trend import *
import cv2
from cnn.multi_stage import msHconvResNet, BiMsHconvResNet
from cnn.xy import StockXY

font_path = '/h"ome/snova/Projects/discrimination/jpy_files/wordcloud/msyhbd.ttf'
myfont = fm.FontProperties(fname='/home/snova/Projects/discrimination/jpy_files/wordcloud/msyhbd.ttf', size=14)


class msStockXY(StockXY):
    def __init__(self, model_name='msStockXY', mode='train'):
        super().__init__(model_name, mode)
        self.device = torch.device('cuda:0')

    def init_loss(self, reverse_sample_prob=[]):
        # 感知损失
        self.ssim_loss = None
        if self.loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss()
            self.loss_netural = torch.nn.CrossEntropyLoss()
        elif self.loss_type == 'focal':
            self.loss = FocalLoss(gamma=2, alpha=reverse_sample_prob)
            self.loss_netural = FocalLoss(gamma=2, alpha=reverse_sample_prob[0])
        else:
            assert 1 == 2, 'Not support percerptual loss type:%s' % self.loss_type

        self.loss = self.loss.to(self.device)
        self.loss_netural = self.loss_netural.to(self.device)
        # self.loss = nn.DataParallel(self.loss).cuda()
        print('Initialize loss...')

    def train(self, train_path, save_model_id, epochs, batch_size, neutral_sampling_prob=1,
              test_path=None, loss_tresh=0.01, EarlyStopping=True, patience=10,
              loadInMemory=False, draw_history=False):

        # 消息记录
        from utils.system_utils import mess2file
        mess2file = functools.partial(mess2file, file_path=os.path.join(PROJECT_PATH, 'train_logs', ),
                                      file_name=str(save_model_id))

        # initialization
        loss_history = []
        min_loss = float('inf')
        break_loop = False
        extra_epoch = 0
        eval_step = int(epochs / 3)
        eval_flag = False
        if test_path:
            test_dataset = self.init_dataset(test_path, self.period, neutral_sampling_prob=1)
            if test_dataset.__len__() > 0:
                eval_flag = True

        weight_path = os.path.join(PROJECT_PATH, "weights/model_%s.pth" % save_model_id)
        print(self.weight_path)
        dataset = self.init_dataset(train_path, self.period, neutral_sampling_prob)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
            shuffle=True
        )

        # 初始化损失
        self.init_loss(dataset.reverse_sample_prob)

        samples_per_epoch = dataset.__len__()
        steps_per_epoch = int(math.ceil(samples_per_epoch / batch_size))
        mess2file('samples_per_epoch:%s' % samples_per_epoch)
        mess2file('steps_per_epoch:%s' % steps_per_epoch)

        # Init epochs
        epochs = int(abs(epochs))

        # Init learning schedule
        if self.schedule_learn:
            # scheduler = lr_scheduler.StepLR(self.optimizer, step_size=int(epochs * 25 / 600), gamma=0.88)
            step_gap = int(epochs / 3)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, [step_gap, step_gap * 2, step_gap * 3], 0.1,
                -1
            )

        # Init Tensorboard
        if draw_history:
            iter_path = os.path.join(PROJECT_PATH, 'iter_history/model_%s' % save_model_id)
            if os.path.exists(iter_path):
                shutil.rmtree(iter_path)
            broadX = SummaryWriter(iter_path)

        start_time = time.time()

        # normal trainning iteration
        for epoch in range(epochs):
            epoch += 1
            ep_start_time = time.time()
            loss_epoch = []
            torch.cuda.empty_cache()
            for step, (x, label) in enumerate(data_loader):
                step += 1
                global_steps = (epoch - 1) * steps_per_epoch + step
                step_start_time = time.time()

                self.model.train()

                # Sample images
                x, label = x.cuda(), label.cuda()

                # ================== Train   ================== #

                # Generate images
                netural_y, pred = self.model(x, return_netural=True)

                # Calculate loss
                torch.cuda.empty_cache()
                loss_var = self.loss(pred, label)
                loss_netural = self.loss_netural(pred, (label > 0).type(torch.cuda.LongTensor))
                loss = loss_var + loss_netural*0.5
                # loss = g_loss.mean()

                # Backward + Optimize
                self.reset_grad()
                torch.cuda.empty_cache()
                loss.backward()
                self.optimizer.step()

                # eval matrix
                correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = stock_accuracy(pred, label,
                                                                                                    self.multi_label)

                # Record loss
                if draw_history:
                    broadX.add_scalar('step_loss', loss.data.item(), global_steps)
                    broadX.add_scalar('step_correct_prob', correct_prob.data.item(), global_steps)
                    broadX.add_scalar('step_var_correct_prob', var_correct_prob.data.item(), global_steps)
                    broadX.add_scalar('step_netraul_mistake_prob', netraul_mistake_prob.data.item(), global_steps)
                    broadX.add_scalar('step_reverse_prob', reverse_prob.data.item(), global_steps)

                # 保存历史loss
                loss_epoch.append(loss.data.item())

                # Print out log info
                step_elapsed_time = time.time() - step_start_time
                if (step + 1) % 20 == 0 or step == 1:
                    print("Step: %d/%d time: %s,  steo_loss:%s" % (
                        step, steps_per_epoch, round(step_elapsed_time, 1),
                        loss.data.item()) + '\r', end='')

            ep_elapsed_time = time.time() - ep_start_time

            # calculate mean loss
            mean_loss = np.array(loss_epoch).mean()

            # 学习率更新
            if self.schedule_learn:
                scheduler.step()

            loss_history.append(mean_loss)

            # Print the progress
            ep_elapsed_time_mins = round(ep_elapsed_time / 60, 2)
            mess2file("Epoch: %d/%d time: %s mins,  mean loss_total:%s" % (
                epoch, epochs, ep_elapsed_time_mins, mean_loss))

            if draw_history and eval_flag and epoch % eval_step == 0:
                broadX.add_scalar('Epoch_loss', mean_loss, epoch)
                torch.cuda.empty_cache()
                mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = self.evaluate(
                    test_dataset, batch_size)
                mess2file('Epoch: %d/%d Eval test set:: mean_loss:%s, correct_prob:%s,'
                          ' var_correct_prob:%s, netraul_mistake_prob:%s,'
                          ' reverse_prob:%s' % (epoch, epochs,
                                                mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob,
                                                reverse_prob))
                broadX.add_scalar('Epoch_eval_correct_prob', correct_prob, epoch)
                broadX.add_scalar('Epoch_eval_var_correct_prob', var_correct_prob, epoch)

                # eval visual
                train_stock_path = os.path.join(PROJECT_PATH, 'data_source/data/tushare_train', '002594.SZ.npz')
                test_stock_path = os.path.join(PROJECT_PATH, 'data_source/data/tushare_test', '000651.SZ.npz')

                _, traingp = self.predict_trend(train_stock_path)
                _, testpg = self.predict_trend(test_stock_path)
                broadX.add_image('train_eval', cv2.imread(traingp), epoch, dataformats='HWC')
                broadX.add_image('test_eval', cv2.imread(testpg), epoch, dataformats='HWC')
                # todo eval dataset
                # eval_acc = 0
                # broadX.add_scalar('eval_acc', eval_acc.data.item(), global_steps)

            # save weights if mean loss of epoch is mininum
            if EarlyStopping:
                if mean_loss < min_loss:
                    # if (epoch + 1) % 30 == 0:
                    # save weights
                    torch.save(self.model.state_dict(), weight_path)
                    min_loss = mean_loss
                    mess2file('Save weight with min loss %s in epoch %s' % (min_loss, epoch))
                    # reset patience epoch
                    extra_epoch = 0

                    if mean_loss <= loss_tresh:
                        break_loop = True

                    if draw_history and eval_flag:
                        torch.cuda.empty_cache()
                        # todo eval dataset
                        # eval_acc = 0
                        # broadX.add_scalar('eval_acc', eval_acc.data.item(), global_steps)

                else:
                    if EarlyStopping:
                        # Loss converged, stop iteration
                        if extra_epoch >= patience:
                            break_loop = True
                        else:
                            extra_epoch += 1
                            mess2file('Loss havent been improved, current min loss:%s, extra_epoch:%s' % (
                                min_loss, extra_epoch))

                if break_loop:
                    mess2file('Loss converged in %s, stop iteration in epoch %s' % (min_loss, epoch))
                    break

        if not EarlyStopping:
            torch.save(self.model.state_dict(), weight_path)
            mess2file('Save weight with min loss %s in final' % mean_loss)
        self.model.load_state_dict(torch.load(self.weight_path), strict=False)

        elaspe_time = time.time() - start_time
        elaspe_time = round(elaspe_time / 3600, 2)
        mess2file('Elasped trainning time: %s hours' % elaspe_time)

        train_result = self.evaluate(dataset, batch_size)
        mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = train_result
        mess2file('Eval train set:: mean_loss:%s, correct_prob:%s,'
                  ' var_correct_prob:%s, netraul_mistake_prob:%s,'
                  ' reverse_prob:%s' % (mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob))
        if eval_flag:
            mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = self.evaluate(test_dataset,
                                                                                                          batch_size)
            mess2file('Eval test set:: mean_loss:%s, correct_prob:%s,'
                      ' var_correct_prob:%s, netraul_mistake_prob:%s,'
                      ' reverse_prob:%s' % (
                          mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob))

        if draw_history:
            broadX.close()
        # clear caches
        gc.collect()
        torch.cuda.empty_cache()

        return train_result


class msStockResXY(msStockXY):
    def __init__(self, model_name='msStockResXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = msHconvResNet(channel_in=self.features_num, class_num=self.num_class).to(self.device)

class BiMsStockResXY(msStockXY):
    def __init__(self, model_name='BiMsStockResXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = BiMsHconvResNet(channel_in=self.features_num, class_num=self.num_class).to(self.device)




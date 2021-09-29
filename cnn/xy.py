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
from cnn.trm import stockBERT, stockBERTtoken
from utils.np_utils import load, save
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from cnn.matrix import stock_accuracy_np
from trend_ana.sts_trend import *
import cv2
from cnn.advence_res import adResNet, mixResBlock
font_path = '/h"ome/snova/Projects/discrimination/jpy_files/wordcloud/msyhbd.ttf'
myfont = fm.FontProperties(fname='/home/snova/Projects/discrimination/jpy_files/wordcloud/msyhbd.ttf', size=14)


class StockXY(BaseModel):
    def __init__(self, model_name='StockXY', mode='train'):
        super().__init__(model_name, mode)
        self.device = torch.device('cuda:0')

    def predict(self, x):

        if self.eval_model:
            self.model.eval()
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        y_pred = self.model(x)
        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)

        return y_pred

    def predict_trend(self, data_path):

        if self.model is None:
            print('Model need to be created first')
            return None

        data = load(data_path)

        result_data = data.copy().sort_values('date').reset_index(drop=True)
        result_data['pred'] = 0

        data = data.sort_values('date').drop('date', axis=1).reset_index(drop=True)

        for i, row in data.iterrows():
            if i + 1 > self.period:
                if self.drop_y:
                    sample = data.loc[i - self.period + 1:i].drop('y', axis=1).to_numpy().T
                else:
                    sample = data.loc[i - self.period + 1:i]
                    sample.loc[i - self.y_radius:i] = self.y_recent_fill
                    sample = sample.to_numpy().T

                sample = np.expand_dims(sample, axis=0)
                samples = sample if i == self.period else np.concatenate([samples, sample], axis=0)

        y_pred = self.predict(samples)

        # format
        result_data.loc[self.period:, 'pred'] = y_pred
        result_data['peak'] = result_data.y == 2
        result_data['bottom'] = result_data.y == 1
        result_data['downward_line'] = result_data.pred == 2
        result_data['upward_line'] = result_data.pred == 1

        # 计算准确度
        window_radius = 6
        acc_pd = result_data.loc[self.period + 1:len(result_data) - window_radius].astype(int)
        sensitivity, specificity, accuracy, trend_acc = cal_trend_pred_acc(acc_pd, true_set_key=('peak', 'bottom'),
                                                                           pred_set_key=(
                                                                               'downward_line', 'upward_line'),
                                                                           _shfit_treshold=0)

        correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = stock_accuracy_np(np.array(acc_pd.pred),
                                                                                               np.array(acc_pd.y))

        correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = round(correct_prob, 3), \
                                                                             round(var_correct_prob, 3), \
                                                                             round(netraul_mistake_prob, 3), \
                                                                             round(reverse_prob, 3)

        # 计算预期收益率
        interest_pd = result_data.loc[self.period + 1:len(result_data) - window_radius]

        final_interes_rate = cal_expected_interest(interest_pd, price_key='close',
                                                   trend_key=('downward_line', 'upward_line'))

        # summary graph
        date = result_data.date.astype(str).tolist()
        x = np.linspace(0, len(result_data.close) - 1, num=len(result_data.close))

        fig, ax = plt.subplots(figsize=(15, 10))
        a1, = ax.plot(x, result_data.close.tolist(), zorder=10)

        col_labels = ['prob']
        row_labels = ['correct', 'var_correct', 'netraul_mistake', 'reverse', 'interest']
        table_vals = [correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob, final_interes_rate]
        table_vals = [[str(int(i * 100)) + str('%')] for i in table_vals]
        table = plt.table(cellText=table_vals,
                          rowLabels=row_labels,
                          rowColours=['tomato', 'palegreen', 'silver', 'skyblue', 'gold'],
                          colColours=['darkgrey'], colLabels=col_labels,
                          loc='lower right',
                          colWidths=[0.1] * 2, zorder=10
                          )

        gl = ax.vlines(result_data[result_data['upward_line']].index, 0, 1,
                       transform=ax.get_xaxis_transform(), colors='r', linestyle='--', zorder=1)
        dl = ax.vlines(result_data[result_data['downward_line']].index, 0, 1,
                       transform=ax.get_xaxis_transform(), colors='g', linestyle='--', zorder=1)

        ax.legend((a1, gl, dl), ('close price', 'trend line', 'buy', 'sell'), fontsize=15, loc='upper left')
        sid = data_path.split('/')[-1].split('.')[0]
        graph_dir = os.path.join(PROJECT_PATH, 'graph/stock_cnn_pred/%s' % (date[0] + '~' + date[-1]))
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        graph_path = os.path.join(graph_dir, '%s.png' % sid)
        plt.savefig(graph_path)

        return result_data, graph_path

    def distribute_model(self):
        self.model = nn.DataParallel(self.model)

    def init_optimizer(self):

        if self.optimizer == 'sgdm':
            #  LR_FACTOR: 0.1
            #   LR_STEP:
            #   - 30
            #   - 60
            #   - 90
            #   OPTIMIZER: sgd
            #   LR: 0.05
            #   WD: 0.0001
            #   MOMENTUM: 0.9
            #   NESTEROV: true
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=self.learning_rate, momentum=0.9,
                                             nesterov=True,
                                             weight_decay=self.weight_decay)
            print('Using sgdm optimizer...')
        elif self.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=self.learning_rate)
            print('Using sgd optimizer...')
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                     self.model.parameters()), lr=self.learning_rate)

        print('Initialize optimizer...')

    def init_loss(self, reverse_sample_prob=[]):
        # 感知损失
        self.ssim_loss = None
        if self.loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss()
        elif self.loss_type == 'focal':
            self.loss = FocalLoss(gamma=2, alpha=reverse_sample_prob)
        else:
            assert 1 == 2, 'Not support percerptual loss type:%s' % self.loss_type

        self.loss = self.loss.to(self.device)
        # self.loss = nn.DataParallel(self.loss).cuda()
        print('Initialize loss...')

    def reset_grad(self):
        self.optimizer.zero_grad()

    def create_models(self, load_weights=False):

        if self.mode == 'train':
            self.init_model()

            self.init_optimizer()

            if load_weights:
                self.model.load_state_dict(torch.load(self.weight_path), strict=False)

        else:

            self.init_model()

            if self.eval_model:
                self.model.eval()

            if load_weights:
                self.model.load_state_dict(torch.load(self.weight_path), strict=False)

    def init_dataset(self, train_path, period=4 * 20, neutral_sampling_prob=0.5):

        return DataSetXY(train_path, period, neutral_sampling_prob,
                         multi_label=self.multi_label, drop_y=self.drop_y, y_radius=self.y_radius,
                         y_recent_fill=self.y_recent_fill)

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
                pred = self.model(x)

                # Calculate loss
                torch.cuda.empty_cache()
                loss = self.loss(pred, label)
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
                # train_stock_path = os.path.join(PROJECT_PATH, 'data_source/data/tushare_train', '002594.SZ.npz')
                # test_stock_path = os.path.join(PROJECT_PATH, 'data_source/data/tushare_test', '000651.SZ.npz')
                #
                # _, traingp = self.predict_trend(train_stock_path)
                # _, testpg = self.predict_trend(test_stock_path)
                # broadX.add_image('train_eval', cv2.imread(traingp), epoch, dataformats='HWC')
                # broadX.add_image('test_eval', cv2.imread(testpg), epoch, dataformats='HWC')
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

class StockMixAdResXY(StockXY):
    def __init__(self, model_name='StockMixAdResXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = mixResBlock(period=self.period, channel_in=self.features_num, block=BasicBlock,
                            layers=[3, 4, 6, 3], num_classes=self.num_class, norm_layer=nn.BatchNorm1d,
                                 drop_prob=0.1).to(self.device)

class StockAdResXY(StockXY):
    def __init__(self, model_name='StockAdResXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = adResNet(period=self.period, channel_in=self.features_num,
                            layers=[3, 4, 6, 3], num_classes=self.num_class, norm_layer=nn.InstanceNorm1d).to(self.device)

class StockResXY(StockXY):
    def __init__(self, model_name='StockResXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = ResNet(channel_in=self.features_num, block=BasicBlock,
                            layers=[3, 4, 6, 3], num_classes=self.num_class, norm_layer=nn.BatchNorm1d).to(self.device)


class hconvResXY(StockXY):
    def __init__(self, model_name='hconvResXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = hconvResNet(channel_in=self.features_num, block=hconvBasicBlock,
                                 layers=[3, 4, 6, 3],
                                 num_classes=self.num_class, norm_layer=nn.InstanceNorm1d).to(self.device)


class StockHRXY(StockXY):
    def __init__(self, model_name='StockHRXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = StockHRNet(input_channel=self.features_num, output_channel=self.num_class).to(self.device)


class stockBERTXY(StockXY):
    def __init__(self, model_name='stockBERTXY', mode='train'):
        super().__init__(model_name, mode)

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = stockBERT(self.features_num, self.period, self.num_class, hidden=64, n_layers=4,
                               attn_heads=4, dropout=0).to(self.device)


class stockBERTmlXY(StockXY):
    def __init__(self, model_name='stockBERTmlXY', mode='train'):
        super().__init__(model_name, mode)
        self.multi_label = True

    def predict(self, x):

        if self.eval_model:
            self.model.eval()
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        y_pred = self.model(x)
        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=2)
        y_pred = y_pred[:, -1]
        return y_pred

    def init_model(self):
        print('Init model:%s...' % self.model_name)
        self.model = stockBERTtoken(self.features_num, self.period, self.num_class, hidden=64, n_layers=4,
                                    attn_heads=4, dropout=0).to(self.device)

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
                # pred = self.model(x)
                token, cls = self.model(x)

                # Calculate loss
                torch.cuda.empty_cache()
                loss_multi = self.loss(token, label)
                loss_cls = self.loss(cls, label[:, -1])
                loss = loss_cls + loss_multi
                # loss = g_loss.mean()

                # Backward + Optimize
                self.reset_grad()
                torch.cuda.empty_cache()
                loss.backward()
                self.optimizer.step()

                # eval matrix
                correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = stock_accuracy(cls, label[:, -1],
                                                                                                    False)

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

    def predict(self, x):

        if self.eval_model:
            self.model.eval()
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        token, cls = self.model(x)
        y_pred = np.argmax(cls.detach().cpu().numpy(), axis=1)

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
                token, cls = self.model(x)
                loss_multi = self.loss(token, label)
                loss_cls = self.loss(cls, label[:, -1])
                loss = loss_cls + loss_multi

                # accumulate outputs
                outputs = torch.cat([outputs, cls]) if i > 0 else cls
                targets = torch.cat([targets, label[:, -1]]) if i > 0 else label[:, -1]
                losses = torch.cat([losses, loss.unsqueeze(dim=0)], dim=0) if i > 0 else loss.unsqueeze(dim=0)

                print("Eval Step: %d/%d" % (i, steps_per_epoch) + '\r', end='')

            mean_loss = round(losses.mean().data.item(), 6)
            correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = stock_accuracy(outputs, targets,
                                                                                                multi_label=False)
            correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = correct_prob.data.item(), \
                                                                                 var_correct_prob.data.item(), \
                                                                                 netraul_mistake_prob.data.item(), \
                                                                                 reverse_prob.data.item()
            correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = round(correct_prob, 3), \
                                                                                 round(var_correct_prob, 3), \
                                                                                 round(netraul_mistake_prob, 3), \
                                                                                 round(reverse_prob, 3)

        return mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob

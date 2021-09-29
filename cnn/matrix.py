import torch.nn as nn
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k class"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def stock_accuracy(output, target, multi_label=False):
    with torch.no_grad():
        if multi_label:
            pred = output.contiguous().view(-1, output.size(2))
            pred = torch.argmax(pred, dim=1)
            target = target.view(-1)
        else:
            pred = torch.argmax(output, dim=1)
        batch_size = target.size(0)
        bi_pred = pred.clone()
        bi_target = target.clone()
        bi_pred[bi_pred != 0] = 1
        bi_target[bi_target != 0] = 1

        var_condition = (target > 0).sum() + 0.01
        # 涨跌错分
        reverse_pred = pred * target
        reverse_prob = (reverse_pred == 2).float().sum() / var_condition

        # 中性错分
        netraul_pred = bi_pred + bi_target
        netraul_correct_prob = (netraul_pred == 0).float().sum() / batch_size
        netraul_mistake_prob = (netraul_pred == 1).float().sum() / batch_size

        # 张跌准确率
        var_correct_prob = ((pred == target) & (target > 0)).float().sum() / var_condition

        # 总体准确率
        correct = (pred == target).float().sum()
        correct_prob = correct / batch_size

        return correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob


def stock_accuracy_np(pred, y):


    batch_size = len(y)
    bi_pred = np.copy(pred)
    bi_target = np.copy(y)
    bi_pred[bi_pred != 0] = 1
    bi_target[bi_target != 0] = 1

    var_condition = (y > 0).sum() + 0.01
    # 涨跌错分
    reverse_pred = pred * y
    reverse_prob = (reverse_pred == 2).sum() / var_condition

    # 中性错分
    netraul_pred = bi_pred + bi_target
    netraul_correct_prob = (netraul_pred == 0).sum() / batch_size
    netraul_mistake_prob = (netraul_pred == 1).sum() / batch_size

    # 张跌准确率
    var_correct_prob = ((pred == y) & (y > 0)).sum() / var_condition

    # 总体准确率
    correct = (pred == y).sum()
    correct_prob = correct / batch_size

    return correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob

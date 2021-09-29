# -*- coding: utf-8 -*-
# !/usr/bin/env python


def rl_main(args):
    import warnings
    # 忽略警告输出
    warnings.filterwarnings("ignore")

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_env
    import torch
    def check_pid(pid):
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    import traceback
    import gc
    from utils.system_utils import PROJECT_PATH, mess2file
    import functools
    mess2file = functools.partial(mess2file, file_path=os.path.join(PROJECT_PATH, 'train_logs/'),
                                  file_name=str(args.rl_model_id))

    import json
    import pandas as pd

    # 记录进程
    # 检查相同的gpu设备进程是否在进行
    # 读取当前gpu精修进程

    current_pid = os.getpid()

    pid_file_path = os.path.join(PROJECT_PATH, 'train_scripts')

    gpu_list = args.gpu_env.split(',')

    # 检测gpu占用
    for device in gpu_list:
        pid_file_name = 'pid_gpu_%s.txt' % device
        if pid_file_name in os.listdir(pid_file_path):
            with open(os.path.join(pid_file_path, pid_file_name), 'r') as f:
                pid = int(f.readlines()[0])
                assert not check_pid(pid), 'gpu_device:%s 被占用,进程%s在进行,当前pid:%s' % (device, pid, current_pid)

    # 写入新的进程id
    for device in gpu_list:
        pid_file_name = 'pid_gpu_%s.txt' % device
        with open(os.path.join(pid_file_path, pid_file_name), 'w') as f:
            f.writelines('%s' % current_pid)
        print('new gpu-%s pid %s' % (device, current_pid))

    def make_model(p, model_id, _mode, load_weights=True):
        import cnn
        model_class = cnn.__dict__[p['use_model']](mode=_mode)
        p['model_id'] = model_id

        model_class.set_hyper_paras(p)

        model_class.create_models(load_weights=load_weights)

        return model_class

    # model parameters dict
    para_path = os.path.join(PROJECT_PATH, 'train_scripts/paras/model_%s.json' % args.model_pid)
    with open(para_path, 'r') as json_f:
        p = json.load(json_f)

    # bn是否在预测时使用固定mean,var
    p['eval_bn'] = args.eval_bn

    # optimizer
    p['optimizer'] = args.optimizer
    p['weight_decay'] = args.weight_decay
    p['learning_rate'] = args.learn_rate

    # path
    train_cn = args.train_cn
    test_cn = args.test_cn
    train_path = os.path.join(PROJECT_PATH, 'data_source/data/', train_cn)
    test_path = os.path.join(PROJECT_PATH, 'data_source/data/', test_cn) if test_cn else None

    train_result = None
    test_result = None
    # main process
    if args.fit_model:
        print('Make model...')
        model_class = make_model(p, args.base_model_id, 'train', not args.relearn)

        print('Train model...')
        train_result = model_class.train(train_path, args.rl_model_id, args.epochs, args.batch_size,
                                         args.neutral_sampling_prob,
                                         test_path=test_path, loss_tresh=args.loss_tresh,
                                         EarlyStopping=args.patience > 0, patience=args.patience,
                                         loadInMemory=args.nloadInMemory, draw_history=args.draw_history)

        del (model_class)
        torch.cuda.empty_cache()

        gc.collect()

    # eval
    if args.eval_samples:
        model_class = make_model(p, args.rl_model_id, 'test', load_weights=True)
        test_dataset = model_class.init_dataset(test_path, model_class.period, neutral_sampling_prob=1)
        test_result = model_class.evaluate(test_dataset, args.batch_size)
        mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob = test_result
        mess2file('Eval test set:: mean_loss:%s, correct_prob:%s,'
                  ' var_correct_prob:%s, netraul_mistake_prob:%s,'
                  ' reverse_prob:%s' % (
                      mean_loss, correct_prob, var_correct_prob, netraul_mistake_prob, reverse_prob))

        torch.cuda.empty_cache()
        gc.collect()

    if train_result and test_result:
        compare_file_path = os.path.join(PROJECT_PATH, 'train_logs', 'xy_compare.csv')
        compare_pd = pd.DataFrame(
            {
                'train_cn': [args.train_cn],
                'test_cn': [args.test_cn],
                'epoch': [args.epochs],
                'pid': [args.model_pid],
                'lrate': [args.learn_rate],
                'weight_decay': [args.weight_decay],
                'optimizer': [args.optimizer],
                'batch_size': [args.batch_size],
                'mean_loss': [train_result[0]],
                'correct_prob': [train_result[1]],
                'var_correct_prob': [train_result[2]],
                'netraul_mistake_prob': [train_result[3]],
                'reverse_prob': [train_result[4]],
                'test_mean_loss': [mean_loss],
                'test_correct_prob': [correct_prob],
                'test_var_correct_prob': [var_correct_prob],
                'test_netraul_mistake_prob': [netraul_mistake_prob],
                'test_reverse_prob': [reverse_prob]
            }
        )
        if os.path.exists(compare_file_path):
            _compare_pd = pd.read_csv(compare_file_path)
            compare_pd = compare_pd.append(_compare_pd)
        compare_pd.to_csv(compare_file_path, index=False)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')

    # Used GPU devicex
    parser.add_argument('-gpu_env', type=str, help='使用的显卡设备号', default='0')

    # Fit and Eval setting
    parser.add_argument('-fm', '--fit_model', help='训练模型', action='store_true')
    parser.add_argument('-es', '--eval_samples', help='测试模型', action='store_true')
    parser.add_argument('-train_cn', '--train_cn', help='样本文件名  路径：data_source/data/', default='tushare_train')
    parser.add_argument('-test_cn', '--test_cn', help='样本文件名  路径：data_source/data/', default='tushare_test')

    # Training setting
    parser.add_argument("-pid", "--model_pid", type=str, default="res",
                        help="基准模型ID")
    parser.add_argument("-bid", "--base_model_id", type=str, default="res_ins",
                        help="基准模型ID")
    parser.add_argument("-rlid", "--rl_model_id", type=str, default="res_ins",
                        help="强化学习模型ID")

    parser.add_argument("-ptc", "--patience", type=int, default=0,
                        help="收敛等待迭代数")
    parser.add_argument("-epo", "--epochs", type=int, default=300,
                        help="迭代次数")
    parser.add_argument("-ltresh", "--loss_tresh", type=float, default=-0.1,
                        help="让收敛停止的最小损失值")

    parser.add_argument("-nprob", "--neutral_sampling_prob", type=float, default=1,
                        help="中性样本取样率")

    parser.add_argument("-lrate", "--learn_rate", type=float, default=0.05,
                        help="学习率")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0,
                        help="学习率衰减")
    parser.add_argument("-optim", "--optimizer", type=str, default="adam",
                        help="优化器")

    parser.add_argument('-nli', '--nloadInMemory',
                        help='不加载训练样本到内存', action='store_false')

    parser.add_argument('-dh', '--draw_history', help='画训练迭代历史图', action='store_true')

    parser.add_argument("-batch", "--batch_size", type=int, default=4000,
                        help="取样数")

    parser.add_argument('-relearn', '--relearn', help='是否重新学习', action='store_true')

    # eval setting
    parser.add_argument('-evalbn', '--eval_bn', help='bn是否在预测时使用固定mean,var', action='store_true')
    args = parser.parse_args()

    # 开启主程序
    if args.rl_model_id:
        rl_main(args)
    else:
        print('请输入强化学习要保存的新模型ID')

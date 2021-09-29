# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os, sys
import logging
import errno
import datetime
import time
import signal

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def mess2file(mess, file_path=PROJECT_PATH + 'logs/', file_name='stock'):
    print(mess)
    mess = time.strftime('%m-%d,%H:%M:%S', time.localtime(time.time())) + "||" + mess + '\n'
    time_mark = datetime.datetime.now().strftime('%Y-%m-%d %H')

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    sub_file_dir = os.path.join(file_path, file_name)
    if not os.path.exists(sub_file_dir):
        os.makedirs(sub_file_dir)

    error_file_mark = '%s.txt' % time_mark
    error_file_path = os.path.join(sub_file_dir, error_file_mark)

    if not os.path.exists(error_file_path):
        with open(error_file_path, mode='w', encoding='utf-8') as f:
            f.writelines(mess)
    else:
        with open(error_file_path, mode='a', encoding='utf-8') as f:
            f.writelines(mess)


def check_pid(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def redirectPrint(file_path=PROJECT_PATH + 'retouch_logs/', file_name='retouch'):
    time_mark = datetime.datetime.now().strftime('%Y-%m-%d %H')
    error_file_mark = '%s_%s.txt' % (file_name, time_mark)
    error_file_path = file_path + error_file_mark

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if not os.path.exists(error_file_path):
        sys.stdout = open(error_file_path, mode='w', encoding='utf-8')
    else:
        sys.stdout = open(error_file_path, mode='a', encoding='utf-8')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def wait_child(signum, frame):
    # 杀死已经结束的僵尸进程
    logging.info('receive SIGCHLD')
    try:
        while True:
            # -1 表示任意子进程
            # os.WNOHANG 表示如果没有可用的需要 wait 退出状态的子进程，立即返回不阻塞
            cpid, status = os.waitpid(-1, os.WNOHANG)
            if cpid == 0:
                logging.info('no child process was immediately available')
                break
            exitcode = status >> 8
            logging.info('child process %s exit with exitcode %s', cpid, exitcode)
    except OSError as e:
        if e.errno == errno.ECHILD:
            logging.warning('current process has no existing unwaited-for child processes.')
        else:
            raise
    logging.info('handle SIGCHLD end')


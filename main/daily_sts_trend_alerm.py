# -*- coding: utf-8 -*-
# !/usr/bin/env python
# # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_DIR)
# sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))
# sys.path.insert(0, os.path.join(BASE_DIR, 'pre_server'))
# sys.path.insert(0, os.path.join(BASE_DIR, 'trend_ana'))
# sys.path.insert(0, os.path.join(BASE_DIR, 'pre_server'))
import json
from utils.time_util import dateutil
import matplotlib.pyplot as plt
from pred_server.mail import MAIL
from utils.sts_util import *
import matplotlib.font_manager as fm
from data_source.data_tushare import get_daily_tushare
from trend_ana.sts_trend import *

font_path = '/home/snova/Projects/discrimination/jpy_files/wordcloud/msyhbd.ttf'
myfont = fm.FontProperties(fname='/home/snova/Projects/discrimination/jpy_files/wordcloud/msyhbd.ttf', size=14)
warnings.filterwarnings("ignore")
dt = dateutil()
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def run_daily_alerm(sid, start_date='20200101', end_date=None,
                    short=11, energy_level_window=10,
                    window_radius=6, sid_dct={}):
    daily = get_daily_tushare(sid, start_date, end_date=end_date)
    len_daily = len(daily)
    if len_daily == 0:
        return None, None, None
    price_data = daily[['date', 'close', 'high', 'low']].sort_values('date').reset_index(drop=True)

    el_data, diff = cal_energy_level(price_data,
                                     high_energy_key='high',
                                     low_energy_key='low',
                                     short=short, energy_level_window=energy_level_window)
    pressure_high = round(el_data.loc[len(el_data) - 1, 'hel_short'] * 10 ** -diff, 1)
    pressure_low = round(el_data.loc[len(el_data) - 1, 'lel_short'] * 10 ** -diff, 1)

    true_trend = cal_true_trend(price_data, grad_key='close',
                                window_radius=window_radius, grad_ma_len=5)

    # 计算准确度
    acc_pd = pd.concat([el_data[['upward_line', 'downward_line']],
                        true_trend[['peak', 'bottom']]],
                       axis=1, join='inner')

    ## 不考虑首尾部数据，因为true值由中心窗口均值计算得出
    acc_pd = acc_pd[short-2+energy_level_window-1+1:-window_radius].astype(int)
    sensitivity, specificity, accuracy, trend_acc = cal_trend_pred_acc(acc_pd, true_set_key=('peak', 'bottom'),
                                                                       pred_set_key=('downward_line', 'upward_line'),
                                                                       _shfit_treshold=10)

    # 计算预期收益率
    interest_pd = pd.concat([el_data[['upward_line', 'downward_line']],
                             price_data[['close']]],
                            axis=1, join='inner')
    interest_pd = interest_pd[short-2+energy_level_window-1+1:-window_radius]

    final_interes_rate = cal_expected_interest(interest_pd, price_key='close',
                                               trend_key=('downward_line', 'upward_line'))

    # summary graph
    date = price_data.date.astype(str).tolist()
    x = np.linspace(0, len(true_trend.close) - 1, num=len(true_trend.close))

    fig, ax = plt.subplots(figsize=(15, 10))
    a1, = ax.plot(x, price_data.close.tolist(), zorder=10)
    a2, = ax.plot(x, true_trend.oma.tolist(), zorder=10)

    col_labels = ['accuracy']
    row_labels = ['uptrend', 'downtrend', 'general', 'trend match', 'interest']
    table_vals = [sensitivity, specificity, accuracy, trend_acc, final_interes_rate]
    table_vals = [[str(int(i * 100)) + str('%')] for i in table_vals]
    table = plt.table(cellText=table_vals,
                      rowLabels=row_labels,
                      rowColours=['tomato', 'palegreen', 'silver', 'skyblue', 'gold'],
                      colColours=['darkgrey'], colLabels=col_labels,
                      loc='lower right',
                      colWidths=[0.1] * 2, zorder=10
                      )

    gl = ax.vlines(el_data[el_data['upward_line']].index, 0, 1,
                   transform=ax.get_xaxis_transform(), colors='r', linestyle='--', zorder=1)
    dl = ax.vlines(el_data[el_data['downward_line']].index, 0, 1,
                   transform=ax.get_xaxis_transform(), colors='g', linestyle='--', zorder=1)

    ax.legend((a1, a2, gl, dl), ('close price', 'trend line', 'buy', 'sell'), fontsize=15, loc='upper left')

    plt.title('股票交易预测：%s' % sid_dct[sid] + '\n时间段: ' + date[0] + '~' + date[-1] +
              '\nclose price:收市价格, tren_line:真实趋势线, buy:预测买入线, sell:预测卖出线' +
              '\nuptrend:买入时机准确率, downtrend:卖出时机准确率, general:交易准确率, trend_match:趋势匹配率, interest:盈利率' +
              '\n压力线:{高位:%s,低位:%s}' % (pressure_high, pressure_low),
              fontsize=100, fontproperties=myfont)

    graph_dir = os.path.join(project_directory, 'graph/stock_base_pred/%s' % (date[0] + '~' + date[-1]))
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    graph_path = os.path.join(graph_dir, '%s.png' % sid_dct[sid])
    plt.savefig(graph_path)

    #last -2 index because there is no right point for comparision of last point
    last_index = el_data.index.tolist()[-2]
    return accuracy, final_interes_rate, graph_path, el_data.loc[last_index, 'upward_line'], el_data.loc[
        last_index, 'downward_line']


def run(sid_file, mail):
    # model parameters dict
    with open('%s/main/paras/%s.json' % (project_directory, sid_file), 'r') as json_f:
        sid_dct = json.load(json_f)

    short = 11
    energy_level_window = 10

    acc_lst = []
    interest_lst = []
    graph_path_lst = []
    sid_lst = []
    name_lst = []
    nodata_lst = []
    u_trend_lst = []
    d_trend_lst = []

    for sid, sname in sid_dct.items():
        result = run_daily_alerm(sid, start_date='20200101',
                                 short=short,
                                 energy_level_window=energy_level_window,
                                 window_radius=6,
                                 sid_dct=sid_dct)

        accuracy, final_interes_rate, graph_path, upward, downward = result
        if accuracy is None:
            nodata_lst.append([sid, sname])
            continue
        if upward or downward:
            acc_lst.append(accuracy)
            interest_lst.append(final_interes_rate)
            graph_path_lst.append(graph_path)
            sid_lst.append(sid)
            name_lst.append(sname)
            u_trend_lst.append(upward)
            d_trend_lst.append(downward)

    print('没有数据列表:%s\n' % str(nodata_lst))
    stock_pd = pd.DataFrame({
        'sid': sid_lst,
        'sname': name_lst,
        'acc': np.around(np.array(acc_lst), 2),
        'interest': np.array(interest_lst),
        'expected_interest': np.around(np.array(acc_lst) * np.array(interest_lst), 3),
        'graph': graph_path_lst,
        'upward': u_trend_lst,
        'downward': d_trend_lst,
    })

    # 计算所有股票，并发送每日前5期望收益率图
    best_pd = stock_pd.sort_values('expected_interest', ascending=False)
    best_pd.expected_interest = (best_pd.expected_interest * 100).astype(str) + '%'

    if len(stock_pd) > 0:
        message_lst = []
        mail_server = MAIL()
        mail_server.toaddr = mail.split(',')
        for i, row in best_pd.iterrows():
            m = '今日趋势：涨\n' if row.upward else '今日趋势：跌\n'
            m += '期望收益率：%s\n股票代码:%s\n 股票名称:%s\n趋势图:\n' % (row.expected_interest, row.sid, row.sname)
            message_lst.append(m)
        title = '今日股票趋势警报'
        mail_server.send_mail_with_img(title, message_lst, best_pd.graph.tolist())


if __name__ == "__main__":

    # import argparse
    #
    # parser = argparse.ArgumentParser(description='manual to this script')
    #
    # # Used GPU devicex
    # parser.add_argument('-sid_file', type=str, help='sid file name', default='rain')
    # parser.add_argument('-mail', type=str, help='mail address lst', default='ldr070@163.com')
    # args = parser.parse_args()

    sid_file_lst = ['rain', 'neng', 'haoyuan','junnan']
    mail_lst = ['ldr070@163.com', 'ldr070@163.com,darrenjliang@hotmail.com',
                'ldr070@163.com,974881911@qq.com', 'ldr070@163.com,1003185084@qq.com']

    for sid_file, mail in zip(sid_file_lst, mail_lst):
        run(sid_file, mail)

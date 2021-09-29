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

sid_dct = {'002594.SZ': '比亚迪', '300474.SZ': '景嘉微', '002371.SZ': '北方华创', '300373.SZ': '杨杰科技', '603986.SH': '兆易创新',
           '300601.SZ': '康泰生物', '000021.SZ': '深科技', '600352.SH': '浙江龙盛', '300207.SZ': '欣旺达', '002475.SZ': '立讯精密',
           '603236.SH': '移远通信',
           '603288.SH': '海天味业', '002709.SZ': '天赐材料', '000661.SZ': '长春高新', '688363.SH': '华熙生物', '603218.SH': '日月股份',
           '600660.SH': '福耀玻璃',
           '000725.SZ': '京东方A', '601012.SH': '隆基股份', '300750.SZ': '宁德时代', '603214.SH': '爱婴室', '600519.SH': '贵州茅台',
           '600336.SH': '澳柯玛', '002732.SZ': '燕塘乳业', '000333.SZ': '美的集团', '000651.SZ': '格力电器', '002617.SZ': '露笑科技',
           '601216.SH': '君正集团',
           '002889.SZ': '东方嘉盛', '002415.SZ': '海康威视', '600887.SH': '伊利股份', '002570.SZ': '贝因美', '601628.SH': '中国人寿',
           '002236.SZ': '大华股份',
           '688012.SH': '中微科技', '002230.SZ': '科大讯飞', '002727.SZ': '一心堂', '600211.SH': '西藏药业'}
# sid_dct = {'002594.SZ': '比亚迪', '601633.SH': '长城汽车', '000625.SZ': '长安汽车'}
# sid_dct = {'600588.SH': '用友网络'}
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def run_daily_pred(sid, start_date='20200101', end_date=None,
                   short=11, energy_level_window=10,
                   window_radius=6, save_flag='upward'):
    daily = get_daily_tushare(sid, start_date, end_date=end_date)
    len_daily = len(daily)
    if len_daily == 0:
        return None, None, None
    price_data = daily[['date', 'close', 'high', 'low']].sort_values('date').reset_index(drop=True)

    el_data, diff = cal_energy_level(price_data,
                                     high_energy_key='high',
                                     low_energy_key='low',
                                     short=short, energy_level_window=energy_level_window)
    pressure_high = round(el_data.loc[len(el_data) - 1, 'hel_short'] * 10 ** -diff,1)
    pressure_low = round(el_data.loc[len(el_data) - 1, 'lel_short'] * 10 ** -diff,1)
    true_trend = cal_true_trend(price_data, grad_key='close',
                                window_radius=window_radius, grad_ma_len=5)

    # 计算准确度
    acc_pd = pd.concat([el_data[['upward_line', 'downward_line']],
                        true_trend[['peak', 'bottom']]],
                       axis=1, join='inner')

    ## 不考虑首尾部数据，因为true值由中心窗口均值计算得出
    acc_pd = acc_pd[short - 1:-window_radius].astype(int)
    sensitivity, specificity, accuracy, trend_acc = cal_trend_pred_acc(acc_pd, true_set_key=('peak', 'bottom'),
                                                                       pred_set_key=('downward_line', 'upward_line'),
                                                                       _shfit_treshold=10)

    # 计算预期收益率
    interest_pd = pd.concat([el_data[['upward_line', 'downward_line']],
                             price_data[['close']]],
                            axis=1, join='inner')
    interest_pd = interest_pd[short - 1:-window_radius]

    final_interes_rate = cal_expected_interest(interest_pd, price_key='close',
                                               trend_key=('downward_line', 'upward_line'))

    # most recently trend
    if el_data[el_data['upward_line']].index[-1] > el_data[el_data['downward_line']].index[-1]:
        now_trend = 'upward'
    else:
        now_trend = 'downward'

    if now_trend == save_flag or save_flag == 'all':
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
                  '\nclose price:开市价格, tren_line:真实趋势线, buy:预测买入线, sell:预测卖出线' +
                  '\nuptrend:买入时机准确率, downtrend:卖出时机准确率, general:交易准确率, trend_match:趋势匹配率, interest:盈利率' +
                  '\n压力线:{高位:%s,低位:%s}' % (pressure_high,pressure_low),
                  fontsize=100, fontproperties=myfont)

        graph_dir = os.path.join(project_directory, 'graph/stock_base_pred/%s' % (date[0] + '~' + date[-1]))
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        graph_path = os.path.join(graph_dir, '%s.png' % sid_dct[sid])
        plt.savefig(graph_path)
    else:
        graph_path = ''

    return accuracy, final_interes_rate, graph_path, now_trend


if __name__ == "__main__":

    only_upward_flag = False
    save_flag = 'upward' if only_upward_flag else 'all'
    mean_interest_lst = []
    short_lst = [11]
    for short in short_lst:
        # el_lst = [x + 2 for x in range(short-1)]
        el_lst = [10]
        for energy_level_window in el_lst:
            print(short, energy_level_window)
            acc_lst = []
            interest_lst = []
            graph_path_lst = []
            sid_lst = []
            name_lst = []
            nodata_lst = []
            now_trend_lst = []
            for sid, sname in sid_dct.items():
                result = run_daily_pred(sid, start_date='20200101',
                                        end_date=None,
                                        short=short,
                                        energy_level_window=energy_level_window,
                                        window_radius=6,
                                        save_flag=save_flag)
                accuracy, final_interes_rate, graph_path, now_trend = result
                if accuracy is None:
                    nodata_lst.append([sid, sname])
                    continue
                acc_lst.append(accuracy)
                interest_lst.append(final_interes_rate)
                graph_path_lst.append(graph_path)
                sid_lst.append(sid)
                name_lst.append(sname)
                now_trend_lst.append(now_trend)
            print('没有数据列表:%s\n' % str(nodata_lst))
            stock_pd = pd.DataFrame({
                'sid': sid_lst,
                'sname': name_lst,
                'acc': np.around(np.array(acc_lst), 2),
                'interest': np.array(interest_lst),
                'expected_interest': np.around(np.array(acc_lst) * np.array(interest_lst), 3),
                'graph': graph_path_lst,
                'now_trend': now_trend_lst
            })

            if only_upward_flag:
                stock_pd = stock_pd[stock_pd.now_trend == 'upward']

            max_acc = stock_pd.acc.max()
            max_acc_pd = stock_pd[stock_pd.acc == max_acc]
            min_acc = stock_pd.acc.min()
            min_acc_pd = stock_pd[stock_pd.acc == min_acc]
            print('最大准确率：%s，股票代码:%s, 股票名称:%s' % (str(max_acc),
                                                 max_acc_pd.sid.values[0],
                                                 max_acc_pd.sname.values[0]))
            print('最低准确率：%s，股票代码:%s, 股票名称:%s' % (str(min_acc),
                                                 min_acc_pd.sid.values[0],
                                                 min_acc_pd.sname.values[0]))

            max_i = stock_pd.interest.max()
            max_i_pd = stock_pd[stock_pd.interest == max_i]
            min_i = stock_pd.interest.min()
            min_i_pd = stock_pd[stock_pd.interest == min_i]
            print('最大收益率：%s，股票代码:%s, 股票名称:%s' % (str(max_i),
                                                 max_i_pd.sid.values[0],
                                                 max_i_pd.sname.values[0]))
            print('最低收益率：%s，股票代码:%s, 股票名称:%s' % (str(min_i),
                                                 min_acc_pd.sid.values[0],
                                                 min_i_pd.sname.values[0]))

            max_ei = stock_pd.expected_interest.max()
            max_ei_pd = stock_pd[stock_pd.expected_interest == max_ei]
            min_ei = stock_pd.expected_interest.min()
            min_ei_pd = stock_pd[stock_pd.expected_interest == min_ei]
            print('最大期望收益率：%s，股票代码:%s, 股票名称:%s' % (str(max_ei),
                                                   max_ei_pd.sid.values[0],
                                                   max_ei_pd.sname.values[0]))
            print('最低期望收益率：%s，股票代码:%s, 股票名称:%s' % (str(min_ei),
                                                   min_ei_pd.sid.values[0],
                                                   min_ei_pd.sname.values[0]))

            mean_acc = np.around(stock_pd.acc.mean(), 2)
            mean_interest = np.around(stock_pd.interest.mean(), 2)
            mean_ei = np.around(stock_pd.expected_interest.mean(), 2)
            mean_interest_lst.append([short, energy_level_window, mean_ei])
            print('\n平均准确率：%s，平均收益率:%s, 平均期望收益率:%s' % (mean_acc, mean_interest, mean_ei))

    best_p = sorted(mean_interest_lst, key=lambda x: x[2], reverse=True)
    print('Best paras:%s' % str(best_p[0]))
    # 计算所有股票，并发送每日前5期望收益率图
    best_pd = stock_pd.sort_values('expected_interest', ascending=False).head(5)
    best_pd.expected_interest = (best_pd.expected_interest * 100).astype(str) + '%'

    message_lst = []
    mail_server = MAIL()
    # mail_server.toaddr += ['974881911@qq.com']
    mail_server.toaddr += ['darrenjliang@hotmail.com']
    for i, row in best_pd.iterrows():
        m = '期望收益率：%s\n股票代码:%s\n 股票名称:%s\n趋势图:\n' % (row.expected_interest, row.sid, row.sname)
        message_lst.append(m)
    title = '今日股票趋势'
    mail_server.send_mail_with_img(title, message_lst, best_pd.graph.tolist())

from utils.sts_util import *


def cal_energy_level(_data,
                     high_energy_key='high',
                     low_energy_key='low',
                     short=11, energy_level_window=10, align=3):
    data = _data.copy()
    data[high_energy_key], diff = align_num_of_digits(data[high_energy_key], align=align)
    data[low_energy_key], diff = align_num_of_digits(data[low_energy_key], align=align)
    data['lma_short'] = np.ceil(cal_ma(data[low_energy_key].tolist(), n=short))
    data['hma_short'] = np.floor(cal_ma(data[high_energy_key].tolist(), n=short))
    data['lel_short'] = np.ceil(cal_el(data['lma_short'].tolist(), n=energy_level_window, level_max=False))
    data['hel_short'] = np.floor(cal_el(data['hma_short'].tolist(), n=energy_level_window))
    data['energy_level'] = (data['lel_short'] + data['hel_short']) / 2

    data.loc[:short-2, 'lma_short'] = None
    data.loc[:short-2, 'hma_short'] = None
    data.loc[:short-2+energy_level_window-1, 'lel_short'] = None
    data.loc[:short-2+energy_level_window-1, 'hel_short'] = None

    # '平均准确率：0.98，平均收益率:1.4, 平均期望收益率:1.37'
    data['h_diff'] = data['hel_short'] - data['hma_short']
    data['l_diff'] = data['lma_short'] - data['lel_short']
    data['upward_line'], data['downward_line'] = intersection_points(data['l_diff'].fillna(0),
                                                                     data['h_diff'].fillna(0), real_cross=True)
    data.loc[:short-2+energy_level_window-1, 'upward_line'] = False
    data.loc[:short-2+energy_level_window-1, 'downward_line'] = False
    # '平均准确率：0.98，平均收益率:1.4, 平均期望收益率:1.37'
    # upward_flag = (data['h_diff'] < data['l_diff'])
    # downward_flag = (data['h_diff'] > data['l_diff'])
    # upward_flag = upward_flag.shift(-1)
    # downward_flag = downward_flag.shift(-1)
    # data['upward_line'] = (upward_flag) & (data['upward_line'])
    # data['downward_line'] = (downward_flag) & (data['downward_line'])

    return data, diff


def get_num_of_digits(num):
    return len(str(int(num)))


def align_num_of_digits(series, align=3):
    num = series.mean()
    n = get_num_of_digits(num)
    diff = int(align - n)
    return series * 10 ** diff, diff


def cal_true_trend(_true_pd, grad_key='open', window_radius=6, grad_ma_len=5, trend_zone=0, align=3):
    true_pd = _true_pd.copy()
    true_pd[grad_key], diff = align_num_of_digits(true_pd[grad_key], align=align)
    true_pd['oma'] = np.around(cal_ma(true_pd[grad_key].tolist(), n=window_radius, central_window=True), 2)
    true_pd['oma_grad'] = cal_ma(np.gradient(true_pd['oma']), n=grad_ma_len, central_window=True)
    true_pd['oma_grad2nd'] = np.gradient(true_pd['oma_grad'])
    true_pd['bottom'], true_pd['peak'] = intersection_points(true_pd['oma_grad'].fillna(0),
                                                             np.zeros(len(true_pd['oma_grad'])), real_cross=True)
    true_pd['oma'] = true_pd['oma'] * 10 ** -diff
    true_pd[grad_key] = true_pd[grad_key] * 10 ** -diff

    if trend_zone > 1:
        trend_order = true_pd[(true_pd['peak']) | (true_pd['bottom'])]
        trend_order_index = trend_order.index.tolist()
        trend_gap = np.diff(np.array(trend_order_index))
        trend_gap = np.concatenate([trend_gap, [None]])
        trend_order['gap'] = trend_gap
        short_trend_index = trend_order[trend_order['gap'] <= trend_zone].index
        true_pd.loc[short_trend_index, ['bottom', 'peak']] = False

        # # 过滤连续趋势
        # trend_order = true_pd[(true_pd['peak']) | (true_pd['bottom'])]
        # lst = trend_order.bottom.astype(int).tolist()
        # duplicate_order = []
        # last_v = None
        # for i, v in enumerate(lst):
        #     if v == last_v:
        #         duplicate_order.append(i)
        #     last_v = v
        # duplicate_index = [trend_order.index[x] for x in duplicate_order]
        # true_pd.loc[duplicate_index, ['bottom', 'peak']] = False

        # 峰值窗口周围比峰值高的点，谷值周围比谷值低的点，区间内增加原线
        window = int(trend_zone / 2)
        for i, row in true_pd.iterrows():
            if i >= window:
                peak_flag = row['peak']
                bottom_flag = row['bottom']
                price = row['close']

                window_pd = true_pd.loc[i - window:i + window]
                if peak_flag:
                    aug_index = window_pd[window_pd['close'] > price].index
                    min_i = aug_index.min()
                    max_i = aug_index.max()
                    if max_i > i:
                        true_pd.loc[i + 1:aug_index.max(), 'peak'] = True
                    if min_i < i:
                        true_pd.loc[min_i:i - 1, 'peak'] = True
                if bottom_flag:
                    aug_index = window_pd[window_pd['close'] < price].index
                    min_i = aug_index.min()
                    max_i = aug_index.max()
                    if max_i > i:
                        true_pd.loc[i + 1:aug_index.max(), 'bottom'] = True
                    if min_i < i:
                        true_pd.loc[min_i:i - 1, 'bottom'] = True

    return true_pd


def cal_trend_pred_acc(acc_pd, true_set_key=('peak', 'bottom'),
                       pred_set_key=('downward_line', 'upward_line'),
                       _shfit_treshold=10):
    tn_key, tp_key = true_set_key
    pn_key, pp_key = pred_set_key
    true_negative = acc_pd[tn_key].tolist()
    pred_negative = acc_pd[pn_key].tolist()
    true_positive = acc_pd[tp_key].tolist()
    pred_positive = acc_pd[pp_key].tolist()
    shfit_treshold = _shfit_treshold + 1
    N = len(acc_pd)

    true_poistive_shift_lst = []
    true_negative_shift_lst = []
    false_positive_shift_lst = []
    false_negative_shift_lst = []

    true_poistive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0

    for i in range(N):

        if true_positive[i]:
            for j in range(shfit_treshold + 1):
                j_shift = i + j
                if j_shift == N:
                    break
                if pred_positive[j_shift]:
                    true_poistive_shift_lst.append(j)
                    true_poistive_count += 1
                    pred_positive[j_shift] = 0
                    break
                if pred_negative[j_shift]:
                    false_negative_shift_lst.append(j)
                    false_negative_count += 1
                    pred_negative[j_shift] = 0
                    break

        if true_negative[i]:
            for j in range(shfit_treshold + 1):
                j_shift = i + j
                if j_shift == N:
                    break
                if pred_negative[j_shift]:
                    true_negative_shift_lst.append(j)
                    true_negative_count += 1
                    pred_negative[j_shift] = 0
                    break
                if pred_positive[j_shift]:
                    false_positive_shift_lst.append(j)
                    false_positive_count += 1
                    pred_positive[j_shift] = 0
                    break

    # confusion mat
    cond_positive = true_poistive_count + false_negative_count
    cond_negative = true_negative_count + false_positive_count
    cond_positive = cond_positive if cond_positive > 0 else 9999
    cond_negative = cond_negative if cond_negative > 0 else 9999
    sensitivity = true_poistive_count / cond_positive
    specificity = true_negative_count / cond_negative
    sensitivity = np.around(sensitivity, 2)
    specificity = np.around(specificity, 2)

    total_pop = cond_positive + cond_negative
    total_pop = total_pop if total_pop > 0 else 9999
    accuracy = (true_poistive_count + true_negative_count) / total_pop
    accuracy = np.around(accuracy, 2)

    # trend order match
    if total_pop > 0:
        true_trend = np.array(acc_pd[tn_key] * -1 + acc_pd[tp_key])
        true_trend = true_trend[true_trend != 0]
        true_trend = np.array([list(g)[0] for key, g in groupby(true_trend)])
        pred_trend = np.array(acc_pd[pn_key] * -1 + acc_pd[pp_key])
        pred_trend = pred_trend[pred_trend != 0]
        pred_trend = np.array([list(g)[0] for key, g in groupby(pred_trend)])

        diff = len(pred_trend) - len(true_trend)
        if diff > 0:
            true_trend = np.concatenate([true_trend, [0] * diff])
            N = len(pred_trend)
        elif diff < 0:
            pred_trend = np.concatenate([pred_trend, [0] * abs(diff)])
            N = len(true_trend)
        else:
            N = len(true_trend)
        trend_acc = np.around((pred_trend * true_trend).sum() / N, 2)
    else:
        trend_acc = 0

    return sensitivity, specificity, accuracy, trend_acc


def cal_expected_interest(interest_pd, price_key='close',
                          trend_key=('downward_line', 'upward_line')):
    sell_line, buy_line = trend_key
    interest_pd[sell_line] = interest_pd[sell_line].shift(1).fillna(False)
    interest_pd[buy_line] = interest_pd[buy_line].shift(1).fillna(False)

    trade_order = interest_pd[(interest_pd[sell_line]) | (interest_pd[buy_line])]

    last_opt = 'sell'
    hold_price = None
    final_interes_rate = 1
    for i, row in trade_order.iterrows():
        sell_flag = row[sell_line]
        buy_flag = row[buy_line]
        price = row[price_key]

        if sell_flag and last_opt == 'buy':
            interest_rate = price / hold_price
            final_interes_rate = final_interes_rate * interest_rate
            hold_price = price
            last_opt = 'sell'
        if buy_flag and last_opt == 'sell':
            hold_price = price
            last_opt = 'buy'

    final_interes_rate = np.around(final_interes_rate, 2)

    return final_interes_rate
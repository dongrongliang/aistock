# -*- coding: utf-8 -*-

import json
from utils.np_utils import *
import tushare as ts
import pandas as pd
import time
from pandas.tseries.offsets import Day
from utils.time_util import dateutil
from trend_ana.sts_trend import cal_energy_level, cal_true_trend
import os
from utils.system_utils import *
from utils.sts_util import *

project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
dt = dateutil()
ts.set_token('8ebd6a1ab14dc2795d9669d457660d684780b2a0f2560b7b501c532c')
pro = ts.pro_api()


def get_daily_index_tushare(str_start_date=None, str_end_date=None):
    str_end_date = str_end_date if str_end_date else dt.getNow()
    extend_start = dt.getLastDays(str_start_date, 7)

    end_date = ''.join(str_end_date.split('-'))
    extend_start = ''.join(extend_start.split('-'))

    # domestic
    domestic_sids = ['000001.SH', '399001.SZ', '399006.SZ']
    df_sh = pro.index_daily(ts_code='000001.SH', start_date=extend_start, end_date=end_date)
    df_sh = df_sh[['trade_date', 'pct_chg']].rename(columns={'pct_chg': 'sh_chg'})
    df_sz = pro.index_daily(ts_code='399001.SZ', start_date=extend_start, end_date=end_date)
    df_sz = df_sz[['trade_date', 'pct_chg']].rename(columns={'pct_chg': 'sz_chg'})
    df_cy = pro.index_daily(ts_code='399006.SZ', start_date=extend_start, end_date=end_date)
    df_cy = df_cy[['trade_date', 'pct_chg']].rename(columns={'pct_chg': 'cy_chg'})

    df_hsi = pro.index_global(ts_code='HSI', start_date=extend_start, end_date=end_date)
    df_hsi = df_hsi[['trade_date', 'pct_chg']].rename(columns={'pct_chg': 'hsi_chg'})

    df_domestic = df_sh.merge(df_sz, on='trade_date', how='left').merge(df_cy, on='trade_date', how='left').merge(
        df_hsi, on='trade_date', how='left')

    df_domestic['date'] = df_domestic['trade_date'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    df_domestic['date'] = pd.to_datetime(df_domestic.date, infer_datetime_format=True)
    df_domestic = df_domestic.drop('trade_date', axis=1).sort_values('date', ascending=True).reset_index(
        drop=True).fillna(method='ffill')

    # global
    global_sids = ['HSI', 'DJI', 'IXIC']
    df_dji = pro.index_global(ts_code='DJI', start_date=extend_start, end_date=end_date)
    df_dji = df_dji[['trade_date', 'pct_chg']].rename(columns={'pct_chg': 'dji_chg'})
    df_ixic = pro.index_global(ts_code='IXIC', start_date=extend_start, end_date=end_date)
    df_ixic = df_ixic[['trade_date', 'pct_chg']].rename(columns={'pct_chg': 'ixic_chg'})

    df_global = df_dji.merge(df_ixic, on='trade_date', how='left')

    df_global['date'] = df_global['trade_date'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    df_global['date'] = pd.to_datetime(df_global.date, infer_datetime_format=True)
    df_global = df_global.drop('trade_date', axis=1)
    df_global['date'] = df_global['date'] + Day(1)
    df_global = df_global.sort_values('date', ascending=True).reset_index(drop=True)

    df = df_domestic.merge(df_global, on='date', how='left').fillna(method='ffill')

    df = df[df.date >= datetime.datetime.strptime(str_start_date, "%Y-%m-%d")]

    assert len(df[df.isnull().T.any()]) == 0, 'daily index null data df:%s' % str(df[df.isnull().T.any()])
    return df

def get_daily_trade_tushare(sid, start_date=None, end_date=None):
    end_date = end_date if end_date else ''.join(dt.getNow().split('-'))
    start_date = start_date if start_date else ''.join(dt.get_n_days(-1).split('-'))

    df = pro.moneyflow(ts_code=sid, start_date=start_date, end_date=end_date)
    df = df[['trade_date', 'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
             'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']]
    df['date'] = df['trade_date'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    df['date'] = pd.to_datetime(df.date, infer_datetime_format=True)
    df = df.drop('trade_date', axis=1)
    return df


def get_daily_tushare(sid, start_date=None, end_date=None):
    end_date = end_date if end_date else ''.join(dt.getNow().split('-'))
    start_date = start_date if start_date else ''.join(dt.get_n_days(-1).split('-'))

    df = pro.daily(ts_code=sid, start_date=start_date, end_date=end_date)

    df['date'] = df['trade_date'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    df['date'] = pd.to_datetime(df.date, infer_datetime_format=True)
    df = df.drop('trade_date', axis=1)
    return df


def get_daily_indicator_tushare(sid, start_date=None, end_date=None):
    end_date = end_date if end_date else ''.join(dt.getNow().split('-'))
    start_date = start_date if start_date else ''.join(dt.get_n_days(-1).split('-'))

    df = pro.daily_basic(ts_code=sid, start_date=start_date, end_date=end_date,
                         fields='ts_code,trade_date,turnover_rate,volume_ratio,pe_ttm,ps_ttm,total_mv')

    df['date'] = df['trade_date'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    df['date'] = pd.to_datetime(df.date, infer_datetime_format=True)
    df = df.drop('trade_date', axis=1)
    return df


def get_finance_tushare(sid, start_date=None, end_date=None, full=False):
    end_date = end_date if end_date else ''.join(dt.getNow().split('-'))
    start_date = start_date if start_date else ''.join(dt.get_n_days(-1).split('-'))

    df = pro.fina_indicator(ts_code=sid, start_date=start_date, end_date=end_date)
    if full:
        df = df[['ts_code', 'ann_date', 'eps', 'profit_to_gr',
                 'basic_eps_yoy', 'op_yoy', 'tr_yoy', 'q_sales_yoy', 'current_ratio',
                 'q_roe',  'ebt_yoy', 'q_op_qoq' ]].drop_duplicates()
    else:
        df = df[['ts_code', 'ann_date', 'eps', 'profit_to_gr',
                 'basic_eps_yoy', 'op_yoy', 'tr_yoy', 'q_sales_yoy', 'current_ratio']].drop_duplicates()

    df['date'] = df['ann_date'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    df['date'] = pd.to_datetime(df.date, infer_datetime_format=True)
    df = df.drop('ann_date', axis=1)
    return df


def get_mat_tushare(sid, period=12 * 28, short=11, energy_level_window=10,
                    window_radius=6, trend_zone=5, full=False, index_data= None):
    begin = dt.get_n_days(-period)
    begin = ''.join(begin.split('-'))

    fina_begin = dt.get_n_days(-period - 4 * 28)
    fina_begin = ''.join(fina_begin.split('-'))

    daily = get_daily_tushare(sid, begin)
    time.sleep(0.2)
    fina = get_finance_tushare(sid, fina_begin, full=full)
    time.sleep(0.2)
    ind = get_daily_indicator_tushare(sid, begin)

    daily = daily.drop(['open', 'change', 'pre_close', 'vol', 'ts_code'], axis=1).sort_values('date').reset_index(
        drop=True)
    fina = fina.drop(['ts_code'], axis=1)
    fina = fina.sort_values('date').reset_index(drop=True)
    ind = ind.drop(['ts_code'], axis=1).sort_values('date').reset_index(drop=True)

    if full:
        time.sleep(0.2)
        trade = get_daily_trade_tushare(sid, begin)
        trade = trade.sort_values('date').reset_index(drop=True)

        for trade_size in ['sm', 'md', 'lg', 'elg']:
            net = trade['buy_%s_amount' % trade_size] - trade['sell_%s_amount' % trade_size]
            ma = cal_ma(net.tolist(), n=11)
            elh = cal_el(net.tolist(), n=10)
            ell = cal_el(net.tolist(), n=10, level_max=False)
            trade['net_%s_diffh' % trade_size] = elh - ma
            trade['net_%s_diffl' % trade_size] = ma - ell

        all_pd = daily.merge(ind, on='date', how='left').merge(trade, on='date', how='left')
        all_pd = all_pd.merge(index_data, on='date', how='left')
    else:
        all_pd = daily.merge(ind, on='date', how='left')

    fina_feat_names = fina.drop('date', axis=1).columns.tolist()
    for feat in fina_feat_names:
        all_pd[feat] = None
    for i, row in all_pd.iterrows():
        d = row.date
        fd = fina[(d - fina.date) >= Day(0)].index
        if len(fd) > 0:
            fina_index = fd[-1]
            for feat in fina_feat_names:
                all_pd.loc[i, feat] = fina.loc[fina_index, feat]
        else:
            for feat in fina_feat_names:
                all_pd.loc[i, feat] = None

    el_data, diff = cal_energy_level(all_pd,
                                     high_energy_key='high',
                                     low_energy_key='low',
                                     short=short, energy_level_window=energy_level_window)

    _all_pd = pd.concat([el_data[['l_diff', 'h_diff']], all_pd], axis=1)
    mat_pd = _all_pd.sort_values('date').drop(['total_mv', 'amount'], axis=1)

    true_pd = cal_true_trend(mat_pd, grad_key='close',
                             window_radius=window_radius, grad_ma_len=5, trend_zone=trend_zone)

    xy_pd = pd.concat([true_pd[['bottom', 'peak']], mat_pd], axis=1)[
            short - 2 + energy_level_window - 1 + 1:-window_radius]
    xy_pd['y'] = 0
    xy_pd['y'][xy_pd.bottom] = 1
    xy_pd['y'][xy_pd.peak] = 2
    xy_pd = xy_pd.drop(['bottom', 'peak'], axis=1)

    len_null = len(xy_pd[xy_pd.isnull().T.any()])
    if len_null > int(period / 4):
        return None
    else:
        xy_pd = xy_pd[~xy_pd.isnull().T.any()]
        # mat = xy_pd.to_numpy().T
        return xy_pd


def get_daily_top_tushare(date_range=10, compare_set='rain', save_fn='top'):
    stock_lst = []
    for i in range(date_range):
        test_date = dt.get_n_days(-i - 1)
        test_date = ''.join(test_date.split('-'))
        df = pro.top_list(trade_date=test_date)
        ts_code_lst = df.ts_code.tolist()
        name_lst = df.name.tolist()
        if len(df) > 0:
            stock_lst.extend(list(zip(ts_code_lst, name_lst)))

    with open('%s/main/paras/%s.json' % (PROJECT_PATH, compare_set), 'r') as json_f:
        sid_dct = json.load(json_f)

    out_set = set(stock_lst) - set(sid_dct.items())
    if len(out_set) > 0:
        save_stock = dict(out_set)

    with open('%s/main/paras/%s.json' % (PROJECT_PATH, save_fn), 'w') as f:
        json.dump(save_stock, f, ensure_ascii=False)


if __name__ == "__main__":

    import random
    import time
    split_mode = 'stock'
    sid_file_lst = ['new_tech']
    train_ratio = 0.8
    test_ratio = 0.2
    sample_name = 'fullnewtech2'
    period = int(2 * 12 * 28)
    model_period = 112
    trend_zone = 5
    full_mode = True

    index_data = get_daily_index_tushare(str_start_date=dt.get_n_days(-period)) if full_mode else None

    if split_mode == 'stock':
        train_dir = os.path.join(project_directory, 'data_source/data/%s_train/' % sample_name)
        test_dir = os.path.join(project_directory, 'data_source/data/%s_test/' % sample_name)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            os.makedirs(test_dir)
        for sid_file in sid_file_lst:
            with open('%s/main/paras/%s.json' % (project_directory, sid_file), 'r') as json_f:
                sid_dct = json.load(json_f)

            for sid, sname in sid_dct.items():
                time.sleep(0.5)
                try:
                    data = get_mat_tushare(sid, period, trend_zone=trend_zone, full=full_mode, index_data=index_data)
                    if data is not None:
                        if random.random() < train_ratio:
                            data_path = os.path.join(train_dir, str(sid))
                            save(data, data_path)
                        else:
                            data_path = os.path.join(test_dir, str(sid))
                            save(data, data_path)
                except:
                    print('Fail to extract:%s,%s' % (sid, sname))
    elif split_mode == 'time':
        train_dir = os.path.join(project_directory, 'data_source/data/%s_time_train/' % sample_name)
        test_dir = os.path.join(project_directory, 'data_source/data/%s_time_test/' % sample_name)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            os.makedirs(test_dir)
        for sid_file in sid_file_lst:
            with open('%s/main/paras/%s.json' % (project_directory, sid_file), 'r') as json_f:
                sid_dct = json.load(json_f)

            for sid, sname in sid_dct.items():
                try:
                    data = get_mat_tushare(sid, period, trend_zone=trend_zone, full=full_mode, index_data=index_data)
                    if data is not None:
                        data = data.sort_values('date').reset_index(drop=True)
                        len_data = len(data)
                        split_index = int(len_data * train_ratio)
                        save(data.loc[:split_index - 1], os.path.join(train_dir, str(sid)))
                        save(data.loc[split_index - 1 - model_period:], os.path.join(test_dir, str(sid)))
                except:
                    print('Fail to extract:%s,%s' % (sid, sname))
    else:
        save_dir = os.path.join(project_directory, 'data_source/data/%s/' % sample_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for sid_file in sid_file_lst:
            with open('%s/main/paras/%s.json' % (project_directory, sid_file), 'r') as json_f:
                sid_dct = json.load(json_f)

            for sid, sname in sid_dct.items():
                try:
                    data = get_mat_tushare(sid, period, trend_zone=trend_zone, full=full_mode, index_data=index_data)
                    if data is not None:
                        data_path = os.path.join(save_dir, str(sid))
                        save(data, data_path)
                except:
                    print('Fail to extract:%s,%s' % (sid, sname))

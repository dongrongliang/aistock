# -*- coding: utf-8 -*-
import os
import sys
import time
from datetime import date, datetime, timedelta
import calendar
from dateutil.relativedelta import relativedelta


class dateutil(object):
    """
    时间相关函数封装
    """

    def __init__(self):
        self.today = datetime.strptime(datetime.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        self.daydelta = timedelta(days=1)
        self.now = datetime.now()

    def getDateLists(self, begin, end):
        interval = self.getInterval(begin, end)
        return [self.getLastDays(begin, -x) for x in range(interval + 1)]

    def getMonthLists(self, begin, end):
        begin_date = datetime.strptime(begin, "%Y-%m").date()
        end_date = datetime.strptime(end, '%Y-%m').date()
        temp = end_date
        month_list = []
        while temp >= begin_date:
            month_list.append(temp.strftime('%Y-%m'))
            temp = self.get_before_month(temp.year, temp.month, 1, 1)
        month_list.reverse()
        return month_list

    def get_n_month_ago_begin(self, begin, n):
        year = int(begin[:4])
        month = int(begin[5:7])
        if n > 0:
            for i in range(0, n):
                month -= 1
                if month == 0:
                    month = 12
                    year -= 1
        else:
            for i in range(0, -n):
                if month == 12:
                    month = 0
                    year += 1
                month += 1
        return date(year, month, 1).strftime('%Y-%m')

    def getLastDays(self, begin, interval):
        """
        :param begin:
        :param interval: 正数是之前几天, 负数是之后几天
        :return:
        """
        start = datetime(int(begin[0:4]), int(begin[5:7]), int(begin[8:10]))
        delta = timedelta(days=1)
        if interval < 0:
            for _ in range(0, -interval):
                start = start + delta
        else:
            for _ in range(0, interval):
                start = start - delta
        return start.strftime("%Y-%m-%d")

    def get_n_days(self, interval=0, flag=0):
        """
        负数,过去的几天
        正数,未来的几天
        :param interval:
        :param flag: 1 返回 timedelta
        :return:
        """
        start = self.today
        if interval < 0:
            for _ in range(0, -interval):
                start = start - self.daydelta
        else:
            for _ in range(0, interval):
                start = start + self.daydelta
        if flag == 1:
            return start
        else:
            return start.strftime("%Y-%m-%d")

    def getNow(self):
        """
        获取当天时间
        :return: 当天时间的字符串
        """
        now = datetime.now()
        return now.strftime("%Y-%m-%d")

    def getWeek(self, begin):
        return datetime(int(begin[0:4]), int(begin[5:7]), int(begin[8:10])).strftime("%w")

    def get_today_before_month(self, n):
        """
        获取之前n个月
        :param n:
        :return:
        """
        year = time.localtime()[0]
        month = time.localtime()[1]
        for i in range(0, n):
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        return year, month

    def get_one_month_ago(self, flag=0):
        """
        返回一个月前的时间, 默认返回格式化时间字符串
        flag 1 返回 datetime
        :param flag:
        :return:
        """
        x = self.today-relativedelta(months=1)
        if flag == 1:
            return x
        else:
            return x.strftime("%Y-%m-%d")

    def get_today_before_month_list(self, n=0):
        year = time.localtime()[0]
        month = time.localtime()[1]
        ret = []
        for i in range(0, n):
            month -= 1
            if month == 0:
                month = 12
                year -= 1
            if month >= 10:
                ret.append(str(year) + "-" + str(month) + "-" + "01")
            else:
                ret.append(str(year) + "-0" + str(month) + "-" + "01")
        return ret

    def get_before_month(self, year, month, day, n):
        for i in range(0, n):
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        day = min(day, calendar.monthrange(year, month)[1])
        return date(year, month, day)

    def getInterval(self, begin, end):
        t1 = datetime(int(begin[0:4]), int(begin[5:7]), int(begin[8:10]))
        t2 = datetime(int(end[0:4]), int(end[5:7]), int(end[8:10]))
        return (t2 - t1).days

    def month_first_day(self, flag=False):
        """
        返回当月第一天
        若为1号,则返回上个月1号
        :param flag:
        :return:
        """
        if self.today.day == 1:
            # 如果当日是月初1号，则返回上月1号
            if self.today.month == 1:
                tmp = date(self.today.year - 1, 12, 1)
            else:
                tmp = date(self.today.year, self.today.month - 1, 1)
        else:
            tmp = date(self.today.year, self.today.month, 1)
        return tmp if flag else tmp.strftime("%Y-%m-%d")

    def get_today(self, flag=False):
        return self.today if flag else self.today.strftime("%Y-%m-%d")

    def get_n_pre_month_first_day(self, n, flag=False):
        """
        获取 n 个月前第一天
        """
        r = self.get_before_month(self.today.year, self.today.month, 1, n)
        return r if flag else r.strftime("%Y-%m-%d")

    def get_n_month_ago(self, n, flag=0):
        d = self.get_before_month(self.today.year, self.today.month, self.today.day, n)
        return d.strftime("%Y-%m-%d") if flag == 0 else d

    def get_week_ago(self, flag=False):
        """
        :param: None
        :return: 7 days ago
        """
        tmp = (self.today - timedelta(days=7))
        return tmp if flag else tmp.strftime("%Y-%m-%d")

    def get_week_first_day(self, flag=False):

        """
        返回当前天的一周开始
        :param flag:
        :return:
        """
        # print self.today.day
        if self.today.weekday() == 0:
            # 如果当天是周一，则返回上周一日期
            tmp = self.today - timedelta(days=7)
        else:
            tmp = self.today - timedelta(days=self.today.weekday())

        return tmp if flag else tmp.strftime("%Y-%m-%d")

    def get_one_day_ago(self, flag=False):
        """
        :param: None
        :return: 1 day ago
        """
        tmp = (self.today - timedelta(days=1))
        return tmp if flag else tmp.strftime("%Y-%m-%d")

    def get_start(self, s_start=date.today().strftime("%Y-%m-%d")):
        return datetime(int(s_start[0:4]), int(s_start[5:7]), int(s_start[8:10]))

    def get_n_hours_ago(self, n=1, flag=1):
        """
        get n hous ago
        flag is True return datetime format
        default 1 hours ago
        if n > 0 :
            过去时间
        else:
            未来时间
        :param n:
        :return:
        """
        r = self.now - timedelta(hours=n)
        return r if flag else r.strftime("%Y-%m-%d %H:00:00")

    def get_n_minutes_ago(self, n=1, string=True):
        """
        get n minutes ago
        default 1 minutes ago
        if n > 0 :
            过去时间
        else:
            未来时间
        :param n:
        :return:
        """
        if string:
            return (self.now - timedelta(minutes=n)).strftime("%Y-%m-%d %H:%M")
        else:
            return self.now - timedelta(minutes=n)

    def get_n_pre_month_last_day(self, n=0, flag=False):
        """
        获取 n 个月前的最后一天
        :param n:
        :param flag:
        :return:
        """
        r = self.get_before_month(self.today.year, self.today.month, 1, n)
        num = calendar.monthrange(r.year, r.month)[1]
        x = r.replace(day=num)
        return x if flag else x.strftime("%Y-%m-%d")

if __name__ == "__main__":
    ut = dateutil()
    end = ut.now.strftime('%Y-%m') + '-01 00:00:00'
    # begin = ut.get_n_month_ago_begin(ut.now.strftime('%Y-%m'), 1) + '-01 00:00:00'
    # print(ut.get_n_pre_month_first_day(0))
    ut.today = date(2018, 1, 1)
    print(ut.month_first_day())

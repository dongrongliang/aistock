# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os
import yagmail


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

SMTP_PASSWD = 'YPBEHXSQJUKPAKEN'
SMTP_PASSWD_2 = 'jnbruaknmhjpcbbi'


class MAIL(object):

    def __init__(self):
        self.user = '304717956@qq.com'
        self.password = SMTP_PASSWD_2
        self.host = 'smtp.qq.com'
        self.port = '465'
        self.toaddr = ['ldr070@163.com']
        if '' in self.toaddr:
            self.toaddr.remove('')

    def send_mail_with_img(self, title, message_lst, img_path_lst):
        yag = yagmail.SMTP(user=self.user, password=self.password, host=self.host,
                           port=self.port,
                           # smtp_ssl=False
                           )
        for admin in self.toaddr:
            contents = []
            for message, img_path in zip(message_lst, img_path_lst):
                contents += ['\n'+'-'*40,
                            message,
                            yagmail.inline(img_path)
                            ]
                # print(contents)
            yag.send(admin, title, contents)

        yag.close()


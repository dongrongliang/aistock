# -*- coding: utf-8 -*-
"""
Created on 2021-06-02 19:36:08
---------
@summary:
---------
@author: snova
"""

import feapder


class FirstSpider(feapder.AirSpider):
    def start_requests(self):
        yield feapder.Request("https://www.baidu.com")

    def parse(self, request, response):
        print(response)


if __name__ == "__main__":
    FirstSpider().start()
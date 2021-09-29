# -*- coding: utf-8 -*-
"""
Created on 2021-06-02 19:43:27
---------
@summary:
---------
@author: snova
"""

import feapder


class SpiderTest(feapder.AirSpider):
    def start_requests(self):
        yield feapder.Request("https://www.baidu.com")

    def parse(self, request, response):
        print(response)


if __name__ == "__main__":
    SpiderTest().start()
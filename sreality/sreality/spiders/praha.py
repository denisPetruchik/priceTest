# -*- coding: utf-8 -*-
import scrapy
from selenium import webdriver
from scrapy.contrib.spiders import CrawlSpider, Rule
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import json


class product_spiderItem(scrapy.Item):
    id = scrapy.Field()
    stitky = scrapy.Field()
    pudorys = scrapy.Field()
    price = scrapy.Field()
    pass

class PrahaSpider(CrawlSpider):
    name = 'praha'
    allowed_domains = ['sreality.cz']
    start_urls = ['https://www.sreality.cz/hledani/prodej/byty/praha']


    def __init__(self):
        CrawlSpider.__init__(self)
        # use any browser you wish
        self.browser = webdriver.Chrome('~/Downloads/chromedriver') 

    def __del__(self):
        self.browser.close()

    def parse(self, response):
        self.browser.get(response.url)
        # let JavaScript Load
        test_element = WebDriverWait(self.browser, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, "ng-scope")))
        for i in self.browser.find_elements_by_class_name('property'): 
            item = product_spiderItem() 
            dod = i.get_attribute('data-dot-data')
            data = json.loads(dod)
            item['id'] = data['id']
            item['stitky'] = data['stitky']
            item['pudorys'] = data['pudorys']
            yield item



        
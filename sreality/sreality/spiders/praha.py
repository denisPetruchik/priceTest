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
    price = scrapy.Field()
    url = scrapy.Field()
    location = scrapy.Field()
    vlastnictvi = scrapy.Field()
    stavba = scrapy.Field()
    navic = scrapy.Field()
    velikost = scrapy.Field()
    plocha = scrapy.Field()
    podlazi = scrapy.Field()
    novostavba = scrapy.Field()
    pass


class PrahaSpider(CrawlSpider):
    name = 'praha'
    allowed_domains = ['sreality.cz']
    start_urls = ['https://www.sreality.cz/hledani/prodej/byty/praha']
    url_strip = 'https://www.sreality.cz/detail/prodej/byt/'
    

    def __init__(self):
        CrawlSpider.__init__(self)
        # use any browser you wish
        self.browser = webdriver.Chrome(
            '/Users/denispetruchik/Downloads/chromedriver')

    def __del__(self):
    	print 'close'
        self.browser.quit()

    def parse(self, response):
        self.browser.get(response.url)
        # let JavaScript Load
        test_element = WebDriverWait(self.browser, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "ng-scope")))

        items = []
        for i in self.browser.find_elements_by_class_name('property'):
            self.parse_byt(i, items)
            
        nextPage = 2
        page_to_parse = 10
        while (nextPage < page_to_parse):
            self.browser.get(response.url + '?strana=' + str(nextPage))
            # let JavaScript Load
            test_element = WebDriverWait(self.browser, 10).until(
                EC.visibility_of_element_located((By.CLASS_NAME, "ng-scope")))
            for i in self.browser.find_elements_by_class_name('property'):
                self.parse_byt(i, items)
            nextPage += 1

        for i in items:
            yield self.parse_details(i['url'], i)          

    def parse_byt(self, i, items):
        item = product_spiderItem()
        dod = i.get_attribute('data-dot-data')
        data = json.loads(dod)
        item['id'] = data.get('id')
        item['stitky'] = data.get('stitky')        
        details_link = i.find_element_by_css_selector('a.images')
        url = details_link.get_attribute('href')
        item['url'] = url.replace(self.url_strip, '')
        items.append(item)

    def parse_details(self, url, i):
        self.browser.get(self.url_strip + url)
        self.browser.implicitly_wait(5)
        test_element = WebDriverWait(self.browser, 5).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "norm-price")))
        i['price'] = self.browser.find_element_by_css_selector('span.norm-price').text
        i['location'] = self.browser.find_element_by_css_selector('span.location-text').text
        #okres&quot;:&quot;praha-10&quot;,&quot;vlastnictvi&quot;:&quot;druzstevni&quot;,&quot;stavba&quot;:&quot;cihlova&quot;,&quot;navic&quot;:&quot;sklep,vytah&quot;,&quot;typ-nemovitosti&quot;:&quot;byty&quot;,&quot;typ-transakce&quot;:&quot;prodej&quot;,&quot;velikost&quot;:&quot;2+1&quot;}">
        dod = self.browser.find_element_by_css_selector('div.buttons').get_attribute('data-dot-data')
        data = json.loads(dod)
        i['vlastnictvi'] = data.get('vlastnictvi','')
        i['stavba'] = data.get('stavba','')
        i['navic'] = data.get('navic', '')
        i['velikost'] = data.get('velikost','')
        i['novostavba'] = 'False'

        addPars = ''
        nextParse = False
        for par in self.browser.find_elements_by_xpath("//strong[@class='param-value']/span[1]"):
            text = par.text
            if nextParse :
                i['plocha'] = text
                nextParse = False
            if text.find(u'Novostavba') > -1:
                i['novostavba'] = 'True'
            if text.find(u'podlaží z celkem') > -1:
                i['podlazi'] = text
                nextParse = True
        return i

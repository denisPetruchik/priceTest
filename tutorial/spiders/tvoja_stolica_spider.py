# -*- coding: utf-8 -*-

import scrapy

def extract_item_content(item, selection_expression, skip_chars=None):
    content = item.css(selection_expression).extract_first()
    if skip_chars and content:
        return content.strip(skip_chars)
    return content

class TvojaStolicaSpider(scrapy.Spider):
    name = "t-s"
    start_urls = ['http://www.t-s.by/buy/flats/']

    def parse(self, response):
        for item in response.css('ul.apart_body li.apart_item'):
            details_page_url = extract_item_content(item, 'div.item_descr h4 a::attr(href)')
            yield response.follow(details_page_url, callback=self.parse_details)

        next_page = response.css('li.arr a.page-lnk::attr(href)').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

    def parse_details(self, response):
        params = {}
        for item in response.css('ul.about_params li.about_param'):
            key = item.css('div.param_name::text').extract_first()
            value = extract_item_content(item, 'div.param_descr::text', '\n\t ')

            if key == u'Микрорайон':
                value = extract_item_content(item, 'div.param_descr a::text', '\t')

            params[key] = value

        params[u'Цена'] = response.css('div.about_price_ye::text').extract_first()
        params[u'Описание'] = response.css('div.about_descr p::text').extract_first()
        return params

Определение цены кв-ры по параметрам
----------------------------------------------------------------
+Парсим сайты с объявлениями, берем оттуда инфу
+Обучаем модель
+Предсказываем цену кв-ры по параметрам
+Определяем переоценные/недооцененные кв-ры


based on https://docs.scrapy.org/en/latest/intro/tutorial.html

* how to run:
scrapy crawl t-s -o result.json
- append to existing file

scrapy crawl t-s -t json --nolog -o - > result.json
- rewrite file after each run

here "t-s" is the name attribute of TvojaStolicaSpider class

* work from shell
scrapy shell 'http://www.t-s.by/buy/flats/'
scrapy shell 'http://www.t-s.by/buy/flats/825559/'
items = response.css('ul.apart_body table.apart_head').extract()
response.css('ul.apart_body table.apart_head')[0].css('div.usd_price::text').extract()
response.css('li.arr a.page-lnk::attr(href)').extract()
response.css('ul.apart_body table.apart_head')[0].css('td.address div.apart-item_body span::text').extract()
response.css('ul.apart_body table.apart_head')[0].css('div.item_desc a::attr(href)').extract()
response.css('ul.about_params li.about_param div.param_descr')[3].css('a::text').extract()

* fix output for russian characters
add property to settings.py:
FEED_EXPORT_ENCODING = 'utf-8'


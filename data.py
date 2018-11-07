import scrapy
from scrapy.crawler import CrawlerProcess

'''
row:
2  - r2_1st_name_1
4  - QF_1st_name_1
5  - r2_1st_score
6  - r2_1st_name_2
8  - SF_1st_name_1
9  - QF_1st_score
10 - r2_2nd_name_1
12 - QF_1st_score
13 - r2_2nd_score
14 - r2_2nd_name_2
16 - F_name_1
17 - SF_1st_score
18 - r2_3rd_name_1
20 - QF_2nd_name_1
21 - r2_3rd_score
22 - r2_3rd_name_2
24 - SF_1nd_name_2
25 - QF_2nd_score
26 - r2_4th_name_1
28 - QF_2nd_name_2
29 - r2_4th_score
30 - r2_4th_name_2
32 - winner_name
33 - F_score
34 - r2_5th_name_1
36 - QF_3rd_name_1
37 - r2_5th_score 
38 - r2_5th_name_2 
40 - SF_2nd_name_1
41 - QF_3rd_score
42 - r2_6th_name_1
44 - QF_3rd_name_2
45 - r2_6th_score
46 - r2_6th_name_2
48 - F_name_2
49 - SF_2nd_score
50 - r2_7th_name_1
52 - QF_4th_name_1
53 - r2_7th_score
54 - r2_7th_name_2
56 - SF_2nd_name_2
57 - QF_4th_score
58 - r2_8th_name_1
60 - QF_4th_name_2
61 - r2_8th_score
62 - r2_8th_name_2
'''
class QuotesSpider(scrapy.Spider):
    name = "quotes"
    data = []
    def start_requests(self):
        # url = 'http://quotes.toscrape.com/'
        url = 'https://bwfbadminton.com/results/2650/yonex-all-england-open/draw/ms'
        tag = getattr(self, 'tag', None)
        if tag is not None:
            url = url + 'tag/' + tag
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        for quote in response.css('div.quote'):
            found = {
                'text': quote.css('span.text::text').extract_first(),
                'author': quote.css('small.author::text').extract_first(),
            }
            self.data.append(found)
            yield found

        next_page = response.css('li.next a::attr(href)').extract_first()
        if next_page is not None:
            found = response.follow(next_page, self.parse)
            self.data.append(found)
            yield found


process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})

spyder = QuotesSpider()
process.crawl(spyder)
process.start()

foo = 5

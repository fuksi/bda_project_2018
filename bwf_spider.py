import re
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings

# Match-row mapping
# 'match': [1st player row, 2nd player row, score row]
matches = {
    'r2_1': [2,6,5],
    'r2_2': [10,14,13],
    'r2_3': [18,22,21],
    'r2_4': [26,30,29],
    'r2_5': [34,38,37],
    'r2_6': [42,46,45],
    'r2_7': [50,54,53],
    'r2_8': [58,62,61],
    'qf_1': [4,12,9],
    'qf_2': [20,28,25],
    'qf_3': [36,44,41],
    'qf_4': [52,60,57],
    'sf_1': [8,24,17],
    'sf_2': [40,56,49],
    'f'   : [16,48,33]
}

class BwfSpider(scrapy.Spider):
    def __init__(self, url, result, **kwargs):
        self.name = 'BwfSpider'
        self.url = url
        self.data = result 
        super(BwfSpider, self).__init__(self.name, **kwargs)

    def start_requests(self):
        yield scrapy.Request(self.url, self.parse)

    def parse_row(self, row, row_type):
        if row_type == 'score':
            row_text = row.css('.draw-score::text').get()
            scores_text = row_text.split(',')
            match_scores = [score_text.strip().split('-') for score_text in scores_text]
            match_scores_simplified = [(1 if int(score[0]) > int(score[1]) else 0) for score in match_scores]
            return match_scores_simplified

        if row_type == 'rank':
            row_text = row.css('.draw-name').get()
            match = re.search('\[(\d)\]', row_text)
            if match is None:
                return 12
            else:
                return int(match[1])
        
        return []

    def parse(self, response):
        tb_rows = response.css('table.tblDrawsSingle tr[id]')
        for match, rows in matches.items():
            [p1_row, p2_row, score_row] = rows
            p1_rank = self.parse_row(tb_rows[p1_row - 1], 'rank')
            p2_rank = self.parse_row(tb_rows[p2_row - 1], 'rank')
            result = self.parse_row(tb_rows[score_row - 1], 'score')
            winner_rank = self.parse_row(tb_rows[score_row - 2], 'rank')
            result_direction = 1 if winner_rank == p1_rank else 0

            self.data[match] = [p1_rank, p2_rank, result, result_direction]

from bwf_spider import BwfSpider
from scrapy.crawler import CrawlerProcess
import numpy as np
import time


def get_bwf_data():
    categoires = ['ms', 'ws', 'md', 'wd', 'xd']
    tournament_urls = [
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3143/perodua-malaysia-masters-2018/results/draw/',
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3170/yonex-sunrise-dr-akhilesh-das-gupta-india-open-2018/results/draw/',
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3150/toyota-thailand-open-2018/results/draw/'
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3151/singapore-open-2018/results/draw/',
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3154/victor-korea-open-2018/results/draw/'
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3337/yonex-sunrise-hong-kong-open-2018/results/draw/'
    ]

    all_tournaments_result = []
    for base_url in tournament_urls:
        categories_urls = [(category, f'{base_url}{category}')
                           for category in categoires]
        tournament_result = {}

        process = CrawlerProcess(
            {'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'})
        for category, url in categories_urls:
            result = {}
            process.crawl(BwfSpider, url=url, result=result)
            tournament_result[category] = result

        process.start()
        all_tournaments_result.append(tournament_result)

    return all_tournaments_result


all_tournaments_result = get_bwf_data()
result = []
for tournament_result in all_tournaments_result:
    for key, value in tournament_result.items():
        for k, v in value.items():
            result.append(v)

transformed_result = []
for match in result:
    spread = match[1] - match[0]
    np_scores = np.array(match[2])

    # Flip 0, 1 if lose
    if match[3] == 0:
        np_scores = 1 - np_scores

    scores = np_scores.tolist()
    score_cat = 1
    if scores == [0, 1, 0]:
        score_cat = 2
    elif scores == [1, 0, 0]:
        score_cat = 3
    elif scores == [0, 1, 1]:
        score_cat = 4
    elif scores == [1, 0, 1]:
        score_cat = 5
    elif scores == [1, 1]:
        score_cat = 6
    print((spread, score_cat))
    transformed_result.append((spread, score_cat))

with open('kristel.txt', 'w') as f:
    for item in transformed_result:
        f.write(f'{item[0]} {item[1]}\n')

raw_data = np.loadtxt('kristel.txt')

mat = np.zeros(shape=(23, 6))
values = np.arange(46, 0, -1)

for v in values:
    for c in range(0, 6):
        for r in range(0, 23):
            if (c+r) == (46-v):
                mat[r][c] = v

tournament_result = []
for i in raw_data:
    r = int(i[0] + 12 - 1)
    c = int(6 - i[1])
    tournament_result.append(mat[r][c])

filename = f'notebook/data/tournament_{time.time()}'
with open(filename, 'w') as f:
    for item in tournament_result:
        f.write(f'{int(item)}\n')

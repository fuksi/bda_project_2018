from bwf_spider import BwfSpider
from scrapy.crawler import CrawlerProcess
import numpy as np

def get_bwf_data():
    categoires = ['ms', 'ws', 'md', 'wd', 'xd']
    tournament_urls = [
        # 'https://bwfbadminton.com/results/2650/yonex-all-england-open/draw/',
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3337/yonex-sunrise-hong-kong-open-2018/results/draw/',
        'https://bwfbadminton.com/results/2335/yonex-all-england-open/draw/'
    ]

    all_tournaments_result = []
    for base_url in tournament_urls:
        categories_urls = [(category, f'{base_url}{category}') for category in categoires]
        tournament_result = {}

        process = CrawlerProcess({ 'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)' })
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
        for k,v in value.items():
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
    if scores == [0,1,0]:
        score_cat = 2
    elif scores == [1,0,0]:
        score_cat = 3
    elif scores == [0,1,1]:
        score_cat = 4
    elif scores == [1,0,1]:
        score_cat = 5
    elif scores == [1,1]:
        score_cat = 6
    print((spread, score_cat))
    transformed_result.append((spread, score_cat))

foo = 5 # add debug here to pause the program

with open('kristel.txt', 'a') as f:
    for item in transformed_result:
        f.write(f'{item[0]} {item[1]}\n')



from bwf_spider import BwfSpider
from scrapy.crawler import CrawlerProcess

def get_bwf_data():
    categoires = ['ms', 'ws', 'md', 'wd', 'xd']
    tournament_urls = [
        'https://bwfbadminton.com/results/2650/yonex-all-england-open/draw/',
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3337/yonex-sunrise-hong-kong-open-2018/results/draw/'
        # 'https://bwfworldtour.bwfbadminton.com/tournament/3337/yonex-sunrise-hong-kong-open-2018/draw/'
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
    win = sum(match[2]) > 1
    if match[3] == 0:
        win = not win
    result = 1 if win else 0
    transformed_result.append((spread, result))

foo = 5 # add debug here to pause the program

with open('kristel.txt', 'w') as f:
    for item in transformed_result:
        f.write(f'{item[0]} {item[1]}\n')



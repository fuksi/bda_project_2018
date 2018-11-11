from bwf_spider import BwfSpider
from scrapy.crawler import CrawlerProcess

def get_bwf_data():
    categoires = ['ms', 'ws', 'md', 'wd', 'xd']
    tournament_urls = [
        'https://bwfbadminton.com/results/2650/yonex-all-england-open/draw/'
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
foo = 5 # add debug here to pause the program


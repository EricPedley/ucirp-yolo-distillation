from icrawler.builtin import GoogleImageCrawler # pip install icrawler

google_crawler = GoogleImageCrawler(storage={'root_dir': 'google_images_3'})
google_crawler.crawl(keyword='amatuer rocket launch footage', max_num=2000)
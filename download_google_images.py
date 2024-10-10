from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': 'google_images_6'})
google_crawler.crawl(keyword='surface to air missile launch', max_num=2000)
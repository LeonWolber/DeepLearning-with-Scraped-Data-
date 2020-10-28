from apscheduler.scheduler.blocking import BlockingScheduler
import requests
import schedule


def spam_kaggle(amount):
    i = 1
    while i < amount:
        requests.get('https://www.kaggle.com/leonwolber/facebook-reviews-trustpilot')
        if i % 10 == 0:
            print(i)
        i += 1

        scheduler = BlockingScheduler()
        scheduler.add_job(spam_kaggle(3000), 'interval', hours=0.5)
        scheduler.start()
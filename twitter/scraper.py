import time
import sys
import os
import configparser
import csv
import pandas as pd
import numpy as np
import tweepy
sys.path.append("..")
from globals import ROOT_DIR

config = configparser.ConfigParser()
config.read('../config.ini')
consumer_key = config['DEFAULT']['TWITTER_CONSUMER_KEY']
consumer_secret_key = config['DEFAULT']['TWITTER_CONSUMER_SECRET_KEY']
access_token = config['DEFAULT']['TWITTER_ACCESS_TOKEN']
access_token_secret = config['DEFAULT']['TWITTER_ACCESS_TOKEN_SECRET']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

data_file = os.path.join(ROOT_DIR, 'data/founta_data.npy')
data = np.load(data_file)
data = data[()]
users = [value['user']['id'] for key, value in data.items()]

# user_data = {}
user_data = np.load(os.path.join(ROOT_DIR, 'data/user_data.npy'))
user_data = user_data[()]
start = time.time()
for i, user_id in enumerate(users):
    if i < 533:
        continue
    print("On user {}".format(i))
    page_count = 0
    #     for pages in tweepy.Cursor(api.user_timeline, id=user_id, count=200).pages():
    try:
        tweets = api.user_timeline(user_id, count=200)
        user_data[user_id] = [tweet._json['text'] for tweet in tweets]  # each user has list of text tweets

    except:
        print("protected user")
    #     for item in pages:
    #               user_data[user_id].append(item._json['text'])

    np.save(os.path.join(ROOT_DIR, 'data/user_data.npy'), user_data)
#     print(time.time() - start)
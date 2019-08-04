import sys
sys.path.append("..")
import configparser
import tweepy
from utils import *

# set up twitter
config = configparser.ConfigParser()
config.read('../config.ini')
path_data = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_DATA'])
path_labels = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_LABELS'])

consumer_key = config['DEFAULT']['TWITTER_CONSUMER_KEY']
consumer_secret_key = config['DEFAULT']['TWITTER_CONSUMER_SECRET_KEY']
access_token = config['DEFAULT']['TWITTER_ACCESS_TOKEN']
access_token_secret = config['DEFAULT']['TWITTER_ACCESS_TOKEN_SECRET']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# data = np.load(os.path.join(ROOT_DIR, 'data/founta_data.npy'))
# data = data[()]


def search_user_followers(data):
    user_followers = np.load(os.path.join(ROOT_DIR, 'data/user_followers_{}.npz'.format(0)))
    user_followers = user_followers['a']
    user_followers = user_followers[()]
    print(len(user_followers))
    count = 0
    save_count = 0
    print("Starting search...")
    for count, (key, value) in enumerate(data.items()):
        if count < 42:
            continue
        screen_name = value['user']['screen_name']
        ids = []
        for page in tweepy.Cursor(api.followers_ids, screen_name=screen_name, count=5000).pages():
            ids.extend(page)
        user_followers[screen_name] = ids
        print("{} {} {}".format(count, screen_name, len(ids)))
        np.savez(os.path.join(ROOT_DIR, 'data/user_followers_{}.npz'.format(save_count)), a=user_followers)

        if count % 10000 == 0:
            np.savez(os.path.join(ROOT_DIR, 'data/user_followers_{}.npz'.format(save_count)), a=user_followers)
            save_count += 1
            print("Saving user_followers_list")
            user_followers = {}

    np.savez(os.path.join(ROOT_DIR, 'data/user_followers_{}.npz'.format(save_count)), a=user_followers)

import time
import string

def search_user_timelines(users):
    save_count = 0
    user_data = {}
    for i, (user_id) in enumerate(users):
        tweets = []
        try:
            for pages in tweepy.Cursor(api.user_timeline, id=user_id, count=200).pages():
                tweets.extend(pages)
                if len(tweets) > 200:
                    break
        except:
            pass

        print("User {} / {} with tweet length {}".format(i, len(users), len(tweets)))
        tweets = [tweet._json['text'] for tweet in tweets]
        tokens = []
        for item in tweets:
            item = item.translate(str.maketrans('', '', string.punctuation)).lower()
            item = item.split(' ')
            tokens.append(item)

        user_data[user_id] = tokens # each user has list of text tweets
        if i % 1000 == 0:
            np.savez(os.path.join(ROOT_DIR, 'data/user_timeline_processed_{}.npz'.format(save_count)), a=user_data)
            user_data = {}
            save_count += 1
    # save excess
    np.savez(os.path.join(ROOT_DIR, 'data/user_timeline_processed_{}.npz'.format(save_count)), a=user_data)

missing = np.load(os.path.join(ROOT_DIR, 'data/missing.npz'), allow_pickle=True)
missing = missing['a']
missing = missing[()]
search_user_timelines(missing)


#
# timelines = {}
# for i in range(66):
#     results = np.load(os.path.join(ROOT_DIR, 'data/user_timeline_{}.npz'.format(i)), allow_pickle=True)
#     print("Downloading User Timelines, Processed {} / {}".format(i+1, 66))
#     results = results['a']
#     results = results[()]
#     timelines = {**results, **timelines}
#
# def clean_timelines(timelines):
#     save_count = 0
#     processed_timelines = {}
#     for i, (key, value) in enumerate(timelines.items()):
#         if i % 10000 == 0 and i != 0:
#             np.savez(os.path.join(ROOT_DIR, 'data/user_timeline_processed_{}.npz'.format(save_count)), a=processed_timelines)
#             save_count += 1
#             processed_timelines = {}
#             print(i)
#         tokens = []
#         for item in value:
#             item = item.translate(str.maketrans('', '', string.punctuation)).lower()
#             item = item.split(' ')
#             tokens.append(item)
#         processed_timelines[key] = tokens
#     return processed_timelines
# clean_timelines(timelines)
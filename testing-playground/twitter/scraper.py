#  start)import time
# # import sys
# # import os
# # import configparser
# # import csv
# # import pandas as pd
# # import numpy as np
# # import tweepy
# # sys.path.append("..")
# # from globals import ROOT_DIR
# #
# # config = configparser.ConfigParser()
# # config.read('../config.ini')
# # consumer_key = config['DEFAULT']['TWITTER_CONSUMER_KEY']
# # consumer_secret_key = config['DEFAULT']['TWITTER_CONSUMER_SECRET_KEY']
# # access_token = config['DEFAULT']['TWITTER_ACCESS_TOKEN']
# # access_token_secret = config['DEFAULT']['TWITTER_ACCESS_TOKEN_SECRET']
# #
# # auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
# # auth.set_access_token(access_token, access_token_secret)
# # api = tweepy.API(auth)
# #
# # data_file = os.path.join(ROOT_DIR, 'data/founta_data.npy')
# # data = np.load(data_file)
# # data = data[()]
# # users = [value['user']['id'] for key, value in data.items()]
# #
# # # user_data = {}
# # user_data = np.load(os.path.join(ROOT_DIR, 'data/user_data.npy'))
# # user_data = user_data[()]
# # start = time.time()
# # for i, user_id in enumerate(users):
# #     if i < 533:
# #         continue
# #     print("On user {}".format(i))
# #     page_count = 0
# #     #     for pages in tweepy.Cursor(api.user_timeline, id=user_id, count=200).pages():
# #     try:
# #         tweets = api.user_timeline(user_id, count=200)
# #         user_data[user_id] = [tweet._json['text'] for tweet in tweets]  # each user has list of text tweets
# #
# #     except:
# #         print("protected user")
# #     #     for item in pages:
# #     #               user_data[user_id].append(item._json['text'])
# #
# #     np.save(os.path.join(ROOT_DIR, 'data/user_data.npy'), user_data)
# # #     print(time.time() -

from bert_embedding import BertEmbedding
import numpy as np
import os
import sys
sys.path.append("..")
from globals import ROOT_DIR
import numpy as np
import os
import sys
sys.path.append("..")
from globals import ROOT_DIR
import time

data = np.load(os.path.join(ROOT_DIR, 'data/reply_data.npy'))
data = data[()]

tweet_text = []
tweet_ids = []
count = 0
for key, value in data.items():
    tweet_ids.append(int(key))
    tweet_text.append(value.lower())
    count += 1


def generate_random_embedding(embed_dim):
    return np.random.normal(scale=0.6, size=(embed_dim,))


def process_embedding(embedding):
    # trim if too large
    if len(embedding) > 17:
        embedding = embedding[:17]

    if len(embedding) < 17:
        diff = 17 - len(embedding)
        embedding += [generate_random_embedding(768) for _ in range(diff)]
    return embedding


start = time.time()
bert_embedding = BertEmbedding()
start_ptr = 0
rate = 100
end_ptr = start_ptr + rate
print("loading file...")
# results = np.load(os.path.join(ROOT_DIR, 'data/bert_embeddings.npz'))
# results = results['arr_0']

results = {}
item_count = start_ptr
save_count = 8
while (start_ptr <= len(tweet_text)):
    if start_ptr % 10000 == 0:
        print("Saving at {}".format(start_ptr))
        np.savez(os.path.join(ROOT_DIR, 'data/bert_embeddings_{}.npz'.format(save_count)), a=results)
        save_count += 1
        results = {}
    raw_results = bert_embedding(tweet_text[start_ptr:end_ptr])
    for item in raw_results:
        embed = process_embedding(item[1])
        results[int(tweet_ids[item_count])] = embed
        item_count += 1
    print("Total items processed: {}".format(item_count))
    start_ptr += rate
    end_ptr += rate

print("Saving final time...")
np.savez(os.path.join(ROOT_DIR, 'data/bert_embeddings_{}'.format(save_count)), a=results)
print("End time: {}".format((time.time() - start) / 60))



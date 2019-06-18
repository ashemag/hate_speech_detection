"""
Helpers for tweet extraction/processing
"""
import csv
import json
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessor import Preprocessor


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    return x_train, y_train, x_val, y_val, x_test, y_test


def extract_labels(filename):
    print("Extracting annotations")
    data = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['tweet_id']] = row['maj_label']
    return data


def generate_random_embedding(embed_dim):
    return np.random.normal(scale=0.6, size=(embed_dim,))


def process_text(text):
    p = Preprocessor()
    p.clean(text)
    p.tokenize()
    return p.text, p.tokens


def extract_tweets(data, filename, subset=None):
    print("Extracting tweets from JSON")
    tweets = []
    labels = []
    line_count = 0
    labels_map = {'hateful': 0, 'abusive': 1, 'normal': 2, 'spam': 3}
    error_count = 0
    tweet_length = []
    retweet_count = []
    favorite_count = []
    followers_count = []
    tweet_char_length = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if subset is not None and line_count >= subset:
                break
            obj = json.loads(line)
            text_raw = obj['text']

            if data[obj['id_str']] not in labels_map:
                error_count += 1
                continue

            labels.append(labels_map[data[obj['id_str']]])
            if int(labels[-1]) == 0:
                retweet_count.append(obj['retweet_count'])
                followers_count.append(obj['user']['followers_count'])
                favorite_count.append(obj['favorite_count'])
            tweets.append(text_raw)
            tweet_char_length.append(len(text_raw))
            tweet_length.append(len(text_raw.split(' ')))
        line_count += 1
    print("Removed {}/{} labels".format(error_count, line_count))
    print("Average tweet length is {} words".format(int(np.mean(tweet_length))))
    print("Average tweet length is {} characters".format(int(np.mean(tweet_char_length))))
    print("Average {} is {}".format('favorite count', int(np.mean(favorite_count))))
    print("Average {} is {}".format('retweet count', int(np.mean(retweet_count))))
    print("Average {} is {}".format('follower count', int(np.median(followers_count))))
    return tweets, labels

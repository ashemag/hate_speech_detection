"""
Helpers for tweet extraction/processing
"""
import csv
import json
from collections import Counter
import os
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessor import Preprocessor


VERBOSE = True


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
    chars = []
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
    locations = []
    word_length = []
    geo = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if subset is not None and line_count >= subset:
                break
            obj = json.loads(line)

            # for key in obj.keys():
            #     print('{} {}'.format(key, obj[key]))
            # exit()
            geo.append(1 if obj['geo'] is not None else 0)
            locations.append(obj['user']['location'])
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
            words = text_raw.split(' ')
            word_length += [len(word) for word in words]
            tweet_length.append(len(words))
            line_count += 1
    cnt = dict(Counter(word_length))

    # for key in sorted(cnt.keys()):
    #     print("word length: {} frequency: {}".format(key, cnt[key]))
    print("Locations: {}".format(Counter(locations).most_common(10)))
    print("Geos: {}".format(Counter(geo)))
    print("Removed {}/{} labels".format(error_count, line_count))
    print("Average tweet length is {} words".format(int(np.mean(tweet_length))))
    print("Average tweet length is {} characters".format(int(np.mean(tweet_char_length))))
    print("Average {} is {}".format('favorite count', int(np.mean(favorite_count))))
    print("Average {} is {}".format('retweet count', int(np.mean(retweet_count))))
    print("Average {} is {}".format('follower count', int(np.median(followers_count))))
    return tweets, labels


def prepare_output_file(filename, output=None, clean_flag=False):
    """

    :param filename:
    :param output: dictionary to write to csv
    :param clean_flag: bool to delete existing dictionary
    :return:
    """
    file_exists = os.path.isfile(filename)
    if clean_flag:
        if file_exists:
            os.remove(filename)
    else:
        if output is None:
            raise ValueError("Please specify output to write to output file.")

        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(output.keys()))
            if not file_exists:
                writer.writeheader()
            if VERBOSE:
                print("Writing to file {0}".format(filename))
                print(output)
            writer.writerow(output)
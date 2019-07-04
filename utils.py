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


def split_data(x, y, seed, verbose=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=seed)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
    if verbose:
        print("[Class %] Hateful in train: {}, hateful in val: {}, hateful in test: {}"
              .format(round(y_train.count(0)/len(y_train), 2),
                      round(y_val.count(0)/len(y_val), 2),
                      round(y_test.count(0)/len(y_test), 2)
                      )
              )
    return x_train, y_train, x_val, y_val, x_test, y_test


def extract_labels(filename):
    print("=== Extracting annotations ===")
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
    print("=== Extracting tweets from JSON ===")
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
   # print("Locations: {}".format(Counter(locations).most_common(10)))
   # print("Geos: {}".format(Counter(geo)))
    print("[Stats] Removed {}/{} labels".format(error_count, line_count))
    print("[Stats] Average tweet length is {} words".format(int(np.mean(tweet_length))))
    print("[Stats] Average tweet length is {} characters".format(int(np.mean(tweet_char_length))))
    print("[Stats] Average {} is {}".format('favorite count', int(np.mean(favorite_count))))
    print("[Stats] Average {} is {}".format('retweet count', int(np.mean(retweet_count))))
    print("[Stats] Average {} is {}".format('follower count', int(np.median(followers_count))))
    return tweets, labels


def prepare_output_file(filename, output=None, file_action_key='a+'):
    """

    :param filename:
    :param output: dictionary to write to csv
    :param clean_flag: bool to delete existing dictionary
    :param file_action_key: w to write or a+ to append to file
    :return:
    """
    file_exists = os.path.isfile(filename)

    if output is None:
        raise ValueError("Please specify output to write to output file.")
    with open(filename, file_action_key) as csvfile:
        for _, values in output.items():
            break

        fieldnames = ['title', 'test_acc', 'test_f_score', 'test_f_score_hateful', 'test_f_score_abusive',
        'num_experiments',
        'valid_f_score_hateful', 'valid_f_score_abusive', 'epoch', 'train_loss', 'train_acc', 'train_f_score',
        'train_f_score_hateful', 'train_precision_hateful', 'train_recall_hateful', 'train_f_score_abusive',
        'train_precision_abusive', 'train_recall_abusive', 'train_f_score_normal', 'train_precision_normal',
        'train_recall_normal', 'train_f_score_spam', 'train_precision_spam', 'train_recall_spam', 'learning_rate',
        'valid_loss', 'valid_acc', 'valid_f_score', 'valid_precision_hateful', 'valid_recall_hateful',
        'valid_precision_abusive', 'valid_recall_abusive', 'valid_f_score_normal', 'valid_precision_normal',
        'valid_recall_normal', 'valid_f_score_spam', 'valid_precision_spam', 'valid_recall_spam', 'test_loss',
        'test_precision_hateful', 'test_recall_hateful',
        'test_precision_abusive', 'test_recall_abusive', 'test_f_score_normal',
        'test_precision_normal', 'test_recall_normal', 'test_f_score_spam', 'test_precision_spam', 'test_recall_spam',
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or file_action_key == 'w':
            writer.writeheader()
        for _, value in output.items():
            writer.writerow(value)

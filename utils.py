"""
Helpers for tweet extraction/processing
"""
import csv
import json
import os
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
import numpy as np


def split_data(x, y, seed, verbose=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
    total = len(x_train) + len(x_valid) + len(x_test)
    if verbose:
        print("[Sizes] Training set: {:.2f}%, Validation set: {:.2f}%, Test set: {:.2f}%".format(
            len(x_train) / float(total) * 100,
            len(x_valid) / float(total) * 100,
            len(x_test) / float(total) * 100))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


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

    print("[Stats] Removed {}/{} labels".format(error_count, line_count))
    print("[Stats] Average tweet length is {} words".format(int(np.mean(tweet_length))))
    print("[Stats] Average tweet length is {} characters".format(int(np.mean(tweet_char_length))))
    print("[Stats] Average {} is {}".format('favorite count', int(np.mean(favorite_count))))
    print("[Stats] Average {} is {}".format('retweet count', int(np.mean(retweet_count))))
    print("[Stats] Average {} is {}".format('follower count', int(np.median(followers_count))))
    return tweets, labels


def prepare_output_file(filename, output=None, file_action_key='a+', aggregate=False):
    """

    :param filename:
    :param output: dictionary to write to csv
    :param clean_flag: bool to delete existing dictionary
    :param file_action_key: w to write or a+ to append to file
    :return:
    """
    file_exists = os.path.isfile(filename)

    if output is None or output == []:
        raise ValueError("Please specify output list to write to output file.")
    with open(filename, file_action_key) as csvfile:
        fieldnames = list(output[0].keys())
        if aggregate:
            fieldnames = ['title', 'epoch', 'test_f_score', 'test_f_score_hateful',
                          'num_experiments', 'test_acc', 'test_f_score_abusive', 'test_recall_hateful',
                          'test_precision', 'valid_precision', 'train_recall', 'test_recall', 'train_precision',
                          'valid_recall', 'test_recall_normal',
                          'test_loss', 'test_f_score_spam', 'test_recall_normal',
                          'test_precision_spam', 'test_precision_abusive',
                          'test_recall_spam', 'test_f_score_normal', 'test_recall_abusive',
                          'test_precision_normal', 'test_precision_hateful',
                          'train_recall_abusive', 'train_f_score_hateful', 'valid_precision_normal',
                          'train_f_score_spam',
                          'valid_f_score', 'valid_precision_abusive', 'learning_rate', 'valid_recall_normal',
                          'train_precision_spam',
                          'train_f_score_normal', 'valid_recall_abusive', 'valid_loss', 'valid_acc',
                          'train_precision_hateful',
                          'train_recall_spam', 'valid_f_score_normal', 'train_recall_normal', 'valid_recall_spam',
                          'valid_recall_hateful', 'train_loss', 'train_f_score', 'train_acc', 'train_recall_hateful',
                          'valid_precision_spam', 'train_precision_abusive', 'train_f_score_abusive',
                          'valid_f_score_spam',
                          'valid_f_score_hateful', 'valid_precision_hateful', 'valid_f_score_abusive',
                          'train_precision_normal'
                          ]

        to_put_first = ['epoch', 'title']
        for item in to_put_first:
            if item in fieldnames:
                fieldnames.remove(item)
                fieldnames.insert(0, item)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists or file_action_key == 'w' or os.path.getsize(filename) == 0:
            writer.writeheader()

        for entry in output:
            writer.writerow(entry)

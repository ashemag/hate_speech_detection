"""
Helpers for tweet extraction/processing
"""
import csv
import json
import os
from collections import OrderedDict

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
        fieldnames = sorted(list(output[0].keys())) # to make sure new dictionaries in diff order work okay
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
                          'train_precision_normal', 'num_epochs'
                          ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists or file_action_key == 'w' or os.path.getsize(filename) == 0:
            writer.writeheader()

        for entry in output:
            writer.writerow(entry)


if __name__ == "__main__":
    print('testing...')
    merge_dict1 = dict([('test_f_score_hateful', 0.2728), ('test_recall_normal', 0.6829), ('test_precision_abusive', 0.6549), ('test_recall', 0.6872), ('test_f_score', 0.7074), ('test_f_score_spam', 0.5285), ('test_precision', 0.7661), ('test_f_score_normal', 0.7728), ('test_recall_spam', 0.7737), ('test_acc', 0.6872), ('test_precision_hateful', 0.2568), ('test_precision_normal', 0.89), ('test_f_score_abusive', 0.6908), ('test_recall_abusive', 0.7315), ('test_precision_spam', 0.4015), ('test_loss', 0.7477769), ('test_recall_hateful', 0.295), ('valid_precision_abusive', 0.6506), ('train_recall_spam', 0.8113), ('valid_precision_spam', 0.3965), ('valid_f_score_spam', 0.5265), ('valid_f_score_abusive', 0.6715), ('train_loss', 0.62086684), ('valid_recall', 0.685), ('train_precision_hateful', 0.7927), ('train_f_score_spam', 0.7776), ('train_f_score', 0.7512), ('train_precision_normal', 0.6447), ('valid_f_score_hateful', 0.2707), ('valid_precision', 0.7628), ('train_f_score_hateful', 0.8066), ('train_f_score_abusive', 0.805), ('valid_f_score', 0.7047), ('learning_rate', 0.0009), ('train_precision_spam', 0.7471), ('valid_precision_hateful', 0.2642), ('valid_recall_abusive', 0.6943), ('valid_f_score_normal', 0.7728), ('train_recall_abusive', 0.7932), ('train_recall_normal', 0.5853), ('valid_recall_normal', 0.6848), ('train_precision', 0.7515), ('valid_precision_normal', 0.8869), ('train_acc', 0.7531), ('epoch', 16), ('train_recall', 0.7531), ('valid_loss', 0.76342475), ('valid_recall_spam', 0.7864), ('train_f_score_normal', 0.613), ('valid_recall_hateful', 0.2826), ('train_precision_abusive', 0.8178), ('train_recall_hateful', 0.8214), ('valid_acc', 0.685), ('seed', 28), ('title', 'CNN_test_twitter_word'), ('num_epochs', 100)])
    print(merge_dict1)
    prepare_output_file(filename='results.csv',
                        output=[merge_dict1])

    merge_dict2 = dict([('test_precision_abusive', 0.7019), ('test_precision_spam', 0.4204), ('test_recall', 0.6929), ('test_f_score_normal', 0.7814), ('test_recall_normal', 0.7078), ('test_recall_spam', 0.7012), ('test_f_score', 0.7135), ('test_precision_normal', 0.8725), ('test_loss', 0.7417837), ('test_precision_hateful', 0.23), ('test_f_score_spam', 0.5253), ('test_f_score_abusive', 0.6926), ('test_acc', 0.6929), ('test_f_score_hateful', 0.293), ('test_recall_hateful', 0.4055), ('test_recall_abusive', 0.6842), ('test_precision', 0.7595), ('train_precision', 0.76), ('valid_precision_abusive', 0.714), ('valid_precision_hateful', 0.2209), ('learning_rate', 0.0009), ('valid_f_score_hateful', 0.2806), ('valid_recall_spam', 0.7153), ('train_f_score_abusive', 0.8132), ('valid_recall', 0.6906), ('valid_f_score_abusive', 0.6968), ('epoch', 16), ('train_precision_spam', 0.7551), ('valid_acc', 0.6906), ('train_recall_hateful', 0.8455), ('train_f_score_hateful', 0.8236), ('valid_recall_abusive', 0.6806), ('train_precision_hateful', 0.8031), ('train_loss', 0.6023533), ('train_recall_spam', 0.781), ('valid_recall_normal', 0.704), ('valid_precision_spam', 0.4159), ('train_f_score', 0.7598), ('valid_f_score', 0.7122), ('train_recall_normal', 0.6189), ('train_precision_normal', 0.6514), ('train_acc', 0.7608), ('valid_precision', 0.7611), ('valid_loss', 0.7475463), ('valid_f_score_spam', 0.5256), ('train_f_score_normal', 0.6344), ('train_f_score_spam', 0.7676), ('valid_recall_hateful', 0.3863), ('valid_precision_normal', 0.8755), ('valid_f_score_normal', 0.7804), ('train_recall_abusive', 0.7993), ('train_precision_abusive', 0.8283), ('train_recall', 0.7608), ('seed', 27), ('title', 'CNN_test_twitter_word'), ('num_epochs', 100)])

    prepare_output_file(filename='results.csv',
                        output=[merge_dict2])
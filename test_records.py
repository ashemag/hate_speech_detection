#!/usr/bin/env python

"""
Verifies that records obtained through Figure 8
crowdsourcing platform have reasonable annotations
"""
import csv
import numpy as np

KEY = 'immigration'
FILENAME = 'data/' + KEY + '_records.csv'
AMOUNT_TO_CHECK = 10


def extract_data():
    reader = csv.DictReader(open(FILENAME, 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)
    return dict_list[0].keys(), dict_list


def analyze(keys, dict_list):
    n = len(dict_list)
    conf_key = 'does_the_reply_tweet_contain_hateful_language_towards_immigrants_is_it_just_generally_abusive_or_is_its_language_normal:confidence'
    classification_key = 'does_the_reply_tweet_contain_hateful_language_towards_immigrants_is_it_just_generally_abusive_or_is_its_language_normal'
    conf_sum = 0
    migrant_count = 0.0
    normal_count = 0.0
    abusive_count = 0.0
    for entry in dict_list:
        conf_sum += float(entry[conf_key])
        migrant_count += 1 if entry[classification_key] == 'migrants' else 0
        normal_count += 1 if entry[classification_key] == 'normal' else 0
        abusive_count += 1 if entry[classification_key] == 'abusive' else 0

    print("Average confidence in classification is {0}%".format(round((conf_sum / n) * 100, 2)))
    print("Amount classified as hate speech is {0}%".format(round((migrant_count / n) * 100, 2)))
    print("Amount classified as abusive is {0}%".format(round((normal_count / n) * 100, 2)))
    print("Amount classified as normal is {0}%".format(round((abusive_count / n) * 100, 2)))

    # output random amount to verify that classification is reasonable
    indices = np.random.choice(n, AMOUNT_TO_CHECK)
    for i in indices:
        print("===")
        rand_dict = dict_list[i]
        print("PARENT: " + rand_dict['parent_tweet'])
        print("REPLY: " + rand_dict['reply_tweet'])
        print("CLASS: " + rand_dict[classification_key])
        print("CONF: " + rand_dict[conf_key])


if __name__ == '__main__':
    # keys, dict_list = extract_data()
    # analyze(keys, dict_list)

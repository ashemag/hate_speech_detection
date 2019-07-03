"""
Unit tests
"""

import unittest
from gensim.models import word2vec, KeyedVectors
from gensim.test.utils import get_tmpfile
from matplotlib import pyplot
from sklearn.decomposition import PCA
import numpy as np
from baseline_experiment import extract_data, LogisticRegressionDataProvider
from data_providers import CNNTextDataProvider

EMBED_DIM = 200
DEFAULT_SEED = 28
FILENAME = 'data/80k_tweets.json'
FILENAME_LABELS = 'data/labels.csv'


class Testing(unittest.TestCase):
    # def test_data(self):
    #     print("\n=== Data Tests ===\n")
    #     x_train, y_train, x_val, y_val, x_test, y_test = extract_data()
    #     print(type(x_train))
    #     print(len(x_train + x_val))
    #
    #     print("SIZES: training set: {}, validation set: {}, test set: {}".format(len(x_train), len(x_val), len(x_test)))
    #
    # def test_logistic_regression(self):
    #     print("\n=== Logistic Regression Tests ===\n")
    #     dp = LogisticRegressionDataProvider()
    #     data = dp.extract(FILENAME, FILENAME_LABELS)
    #
    def test_CNN(self):
        p = CNNTextDataProvider()
        x_train, y_train, x_val, y_val, x_test, y_test = p.extract(FILENAME, FILENAME_LABELS, key='twitter')


    #
    # def test_vocabulary(self, pretrained_flag=True, saved_flag=True, PCA_flag=False):
    #     print("\n=== Vocabulary Tests ===\n")
    #     p = CNNTextDataProvider()
    #     data = p._extract_labels('data/labels.csv')
    #     tweets, labels = p._extract_tweets(data, 'data/80k_tweets.json', None)
    #     tweets_corpus = p._split_corpus(tweets, labels)
    #     print("Corpus length {}".format(len(tweets_corpus)))
    #
    #     if pretrained_flag:
    #         filename = 'data/GoogleNews-vectors-negative300.bin'
    #         filename = 'data/word2vec_twitter_model/word2vec_twitter_model.bin'
    #         word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
    #     else:
    #         filename = 'data/keyedvectors.bin'
    #         if not saved_flag:
    #             model = word2vec.Word2Vec(sentences=tweets_corpus, size=EMBED_DIM)
    #             model.train(tweets_corpus, total_examples=len(tweets_corpus), epochs=100)
    #             word_vectors = model.wv
    #             word_vectors.save(filename)
    #         word_vectors = KeyedVectors.load(filename)
    #
    #     print("EMBEDDING SIZE {}".format(len(word_vectors['hey'])))
    #     self.assertTrue(len(word_vectors['hey']) > 0)
    #     word_vector_count, random_count = 0, 0
    #     for i, tweet in enumerate(tweets_corpus):
    #         for word in tweet:
    #             if word not in word_vectors:
    #                 random_count += 1
    #             else:
    #                 word_vector_count += 1
    #     print("Random Embedding Count: {} \nWord Vector Embedding Count: {}".format(random_count, word_vector_count))


if __name__ == '__main__':
    unittest.main()

import json
import csv
import time

import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from gensim.models import word2vec, KeyedVectors

from globals import ROOT_DIR
from preprocessor import Preprocessor
import torch.utils.data as data
from utils import *
import pandas as pd
import torch
import torch.utils.data
import torchvision

GOOGLE_EMBED_DIM = 300
TWITTER_EMBED_DIM = 400
TWEET_SENTENCE_SIZE = 17  # 16 is average tweet token length
TWEET_WORD_SIZE = 20 # selected by histogram of tweet counts
FASTTEXT_EMBED_DIM = 300
EMBED_DIM = 200
NUM_CLASSES = 4


class DataProvider(data.Dataset):
    """Generic data provider."""

    def __init__(self, inputs, targets, seed, make_one_hot=False):
        self.inputs = np.array(inputs)
        self.num_classes = len(set(targets))

        if make_one_hot:
            self.targets = self.to_one_of_k(targets)
        else:
            self.targets = np.array(targets)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            inputs, label = dataset[idx]
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[dataset[idx][1]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class TextDataProvider(object):
    def __init__(self, path_data, path_labels):
        self.path_data = path_data
        self.path_labels = path_labels
        self.data = extract_labels(self.path_labels)
        self.raw_tweets, self.labels = extract_tweets(self.data, self.path_data)

    @staticmethod
    def _fetch_model(tweets_corpus, key):
        if key == 'google':
            print("[Model] Using {} embeddings".format(key))
            embed_dim = GOOGLE_EMBED_DIM
            filename = os.path.join(ROOT_DIR, 'data/GoogleNews-vectors-negative300.bin')
            word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        elif key == 'twitter':
            print("[Model] Using {} embeddings".format(key))
            embed_dim = TWITTER_EMBED_DIM
            filename = os.path.join(ROOT_DIR, 'data/word2vec_twitter_model/word2vec_twitter_model.bin')
            word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        elif key == 'fasttext':
            print("[Model] Using {} embeddings".format(key))
            embed_dim = FASTTEXT_EMBED_DIM
            filename = os.path.join(ROOT_DIR, 'data/wiki-news-300d-1M-subword.vec')
            word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        else:
            print("[Model] Using {} embeddings".format(key))
            embed_dim = EMBED_DIM
            model = word2vec.Word2Vec(sentences=tweets_corpus, size=embed_dim)
            model.train(tweets_corpus, total_examples=len(tweets_corpus), epochs=100)
            word_vectors = model.wv
        return word_vectors, embed_dim

    @staticmethod
    def tokenize(raw_tweets):
        tweets = []
        for tweet in raw_tweets:
            text, tokens = process_text(tweet)
            tweets.append(tokens)
        return tweets

    @staticmethod
    def fetch_character_symbols(raw_tweets):
        """
        Dynamically create mapping for one hot encoding of chars
        :param raw_tweets: list of all tweets
        :return:
        """
        chars = set()
        for i, tweet in enumerate(raw_tweets):
            for word in tweet:
                chars.update(list(word))
        char_mapping = {char: np.eye(len(chars))[index] for index, char in enumerate(chars)}
        return chars, char_mapping

    def fetch_character_embeddings(self, raw_tweets):
        print("=== Creating Character Embeddings ===")
        start = time.time()
        char_mapping = self.fetch_character_symbols(raw_tweets)
        char_dimension = len(char_mapping)
        processed_tweets = []
        for i, tweet in enumerate(raw_tweets):
            embedded_tweet = []

            # trim if too large
            if len(tweet) >= TWEET_SENTENCE_SIZE:
                tweet = tweet[:TWEET_SENTENCE_SIZE]

            # convert all into word embeddings
            for word in tweet:
                # trim if too large
                if len(word) >= TWEET_WORD_SIZE:
                    word = word[:TWEET_WORD_SIZE]

                embedded_word = [char_mapping[char] if char in char_mapping else np.zeros(char_dimension) for char in word]

                # pad if too short
                if len(word) <= TWEET_WORD_SIZE:
                    diff = TWEET_WORD_SIZE - len(word)
                    embedded_word += [np.zeros(char_dimension) for _ in range(diff)]

                embedded_tweet.append(np.array(embedded_word))

            # pad if too short
            if len(tweet) <= TWEET_SENTENCE_SIZE:
                diff = TWEET_SENTENCE_SIZE - len(tweet)
                random_words = np.zeros((diff, TWEET_WORD_SIZE, char_dimension))
                for random_word in random_words:
                    embedded_tweet.append(random_word)

            processed_tweets.append(np.array(embedded_tweet))

        print("=== Finished Character Embeddings ({} mins) ===".format(round((time.time() - start) / 60, 2)))
        return processed_tweets

    @staticmethod
    def fetch_word_embeddings(raw_tweets, word_vectors, embed_dim):
        processed_tweets = []
        for i, tweet in enumerate(raw_tweets):
            embedded_tweet = []

            # trim if too large
            if len(tweet) >= TWEET_SENTENCE_SIZE:
                tweet = tweet[:TWEET_SENTENCE_SIZE]

            # convert all into word embeddings
            for word in tweet:
                embedding = generate_random_embedding(embed_dim) if word not in word_vectors else word_vectors[word]
                embedded_tweet.append(embedding)

            # pad if too short
            if len(tweet) < TWEET_SENTENCE_SIZE:
                diff = TWEET_SENTENCE_SIZE - len(tweet)
                embedded_tweet += [generate_random_embedding(embed_dim) for _ in range(diff)]

            assert len(embedded_tweet) == TWEET_SENTENCE_SIZE
            processed_tweets.append(embedded_tweet)
        return processed_tweets

    def generate_tdidf_embeddings(self, seed, verbose=True):
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(self.raw_tweets, self.labels, seed)

        vectorizer = TfidfVectorizer(use_idf=True, max_features=10000, stop_words='english')

        output = {'x_train': vectorizer.fit_transform(x_train).todense(),
                  'y_train': y_train,
                  'x_valid': vectorizer.transform(x_val).todense(),
                  'y_valid': y_val,
                  'x_test': vectorizer.transform(x_test).todense(),
                  'y_test': y_test}

        total = len(output['x_train']) + len(output['x_valid']) + len(output['x_test'])
        if verbose:
            print("[Sizes] Training set: {:.2f}%, Validation set: {:.2f}%, Test set: {:.2f}%".format(
                len(output['x_train']) / float(total) * 100,
                len(output['x_valid']) / float(total) * 100,
                len(output['x_test']) / float(total) * 100))
        return output

    def _generate_embedding_output(self, processed_tweets, seed):
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(processed_tweets, self.labels, seed)
        return {'x_train': x_train,
                'y_train': y_train,
                'x_valid': x_val,
                'y_valid': y_val,
                'x_test': x_test,
                'y_test': y_test
                }

    def generate_word_level_embeddings(self, embedding_key, seed):
        raw_tweets = self.tokenize(self.raw_tweets)
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(raw_tweets, self.labels, seed)
        word_vectors, embed_dim = self._fetch_model(x_train, embedding_key)
        processed_tweets = self.fetch_word_embeddings(raw_tweets, word_vectors, embed_dim)
        return self._generate_embedding_output(processed_tweets, seed)

    def generate_char_level_embeddings(self, seed):
        raw_tweets = self.tokenize(self.raw_tweets)
        processed_tweets = self.fetch_character_embeddings(raw_tweets)
        return self._generate_embedding_output(processed_tweets, seed)

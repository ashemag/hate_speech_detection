import json
import csv
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from gensim.models import word2vec, KeyedVectors
from preprocessor import Preprocessor
import torch.utils.data as data
from utils import *
import pandas as pd

GOOGLE_EMBED_DIM = 300
TWITTER_EMBED_DIM = 400
TWEET_SENTENCE_SIZE = 17  # 16 is average tweet token length
TWEET_WORD_SIZE = 20 # selected by histogram of tweet counts
FASTTEXT_EMBED_DIM = 300
EMBED_DIM = 200
NUM_CLASSES = 4


class DataProvider(data.Dataset):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, seed=28, make_one_hot=True, with_replacement=False):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            seed: to set random state
        """
        self.with_replacement = with_replacement

        self.inputs = inputs
        self.num_classes = len(set(targets))

        if make_one_hot:
            self.targets = self.to_one_of_k(targets)
        else:
            self.targets = targets

        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        self.rng = np.random.RandomState(seed)
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.inputs[index], self.targets[index]

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        if self.with_replacement:
            return self.next_with_replacement()

        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

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

    def next_with_replacement(self):
        self.shuffle()
        batch_slice = slice(self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        return inputs_batch, targets_batch

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch


class LogisticRegressionDataProvider(object):
    def extract(self, filename_data, filename_labels):
        data = extract_labels(filename_labels)
        raw_tweets, labels = extract_tweets(data, filename_data)
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(raw_tweets, labels)

        vectorizer = TfidfVectorizer(use_idf=True, max_features=10000, stop_words='english')
        x_tfidf_train = vectorizer.fit_transform(x_train).todense()
        x_tfidf_val = vectorizer.transform(x_val).todense()
        x_tfidf_test = vectorizer.transform(x_test).todense()
        return x_tfidf_train, y_train, x_tfidf_val, y_val, x_tfidf_test, y_test


class CNNTextDataProvider(object):
    @staticmethod
    def _fetch_model(tweets_corpus, key, saved_flag=True):
        print("Using {} embeddings".format(key))
        if key == 'google':
            embed_dim = GOOGLE_EMBED_DIM
            filename = 'data/GoogleNews-vectors-negative300.bin'
            word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        elif key == 'twitter':
            embed_dim = TWITTER_EMBED_DIM
            filename = 'data/word2vec_twitter_model/word2vec_twitter_model.bin'
            word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        elif key == 'fastttext':
            filename = 'data/wiki-news-300d-1M-subword.vec'
            word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        else:
            filename = 'data/keyedvectors.bin'
            embed_dim = EMBED_DIM
            if not saved_flag:
                model = word2vec.Word2Vec(sentences=tweets_corpus, size=embed_dim)
                model.train(tweets_corpus, total_examples=len(tweets_corpus), epochs=100)
                word_vectors = model.wv
                word_vectors.save(filename)
            else:
                word_vectors = KeyedVectors.load(filename)
        return word_vectors, embed_dim

    @staticmethod
    def tokenize(raw_tweets):
        tweets = []
        for tweet in raw_tweets:
            text, tokens = process_text(tweet)
            tweets.append(tokens)
        return tweets

    @staticmethod
    def fetch_character_embeddings(raw_tweets):
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}'
        char_mapping = {char: np.eye(len(chars))[index] for index, char in enumerate(chars)}
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

                embedded_word = [char_mapping[char] if char in char_mapping else np.zeros(len(chars)) for char in word]

                # pad if too short
                if len(word) <= TWEET_WORD_SIZE:
                    diff = TWEET_WORD_SIZE - len(word)
                    embedded_word += [np.zeros(len(chars)) for _ in range(diff)]

                embedded_tweet.append(np.array(embedded_word))

            # pad if too short
            if len(tweet) <= TWEET_SENTENCE_SIZE:
                diff = TWEET_SENTENCE_SIZE - len(tweet)
                random_words = np.zeros((diff, TWEET_WORD_SIZE, len(chars)))
                for random_word in random_words:
                    embedded_tweet.append(random_word)

            processed_tweets.append(np.array(embedded_tweet))

        print("embedded doc shape {}".format(np.array(processed_tweets).shape))
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

    def extract(self, filename_data, filename_labels, embedding_key, embedding_level_key, subset=None):
        data = extract_labels(filename_labels)
        raw_tweets, labels = extract_tweets(data, filename_data, subset)

        if embedding_level_key == 'word':
            raw_tweets = self.tokenize(raw_tweets)
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(raw_tweets, labels)
            print("Hateful tweets in train set are {} of {}".format(y_train.count(0), len(y_train)))
            word_vectors, embed_dim = self._fetch_model(x_train, embedding_key)
            processed_tweets = self.fetch_word_embeddings(raw_tweets, word_vectors, embed_dim)
        else: # CHAR
            raw_tweets = self.tokenize(raw_tweets)
            processed_tweets = self.fetch_character_embeddings(raw_tweets)
        return split_data(processed_tweets, labels)


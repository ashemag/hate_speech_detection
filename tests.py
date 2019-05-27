import unittest

from gensim.models import word2vec

from baseline_experiment import extract_data
from data_provider import TextDataProvider

EMBED_DIM = 200
DEFAULT_SEED = 28


class Testing(unittest.TestCase):
    def test_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = extract_data()
        print(type(x_train))
        print(len(x_train + x_val))

        print("SIZES: training set: {}, validation set: {}, test set: {}".format(len(x_train), len(x_val), len(x_test)))

    def test_vocabulary(self):
        p = TextDataProvider()
        data = p._extract_labels('data/labels.csv')
        _, _ = p._extract_tweets(data, 'data/80k_tweets.json', None)
        tweets_corpus = p._split_corpus()
        print("Corpus length {}".format(len(tweets_corpus)))

        model = word2vec.Word2Vec(sentences=tweets_corpus, size=EMBED_DIM)
        model.train(tweets_corpus, total_examples=len(tweets_corpus), epochs=100)
        word_vectors = list(model.wv.vocab)
        word_vector_dict = model.wv.vocab
        self.assertTrue(len(word_vectors) > 0)
        word_vector_count, random_count = 0, 0

        for i, tweet in enumerate(tweets_corpus):
            for word in tweet:
                if word not in word_vectors:
                    random_count += 1
                else:
                    word_vector_count += 1
                    _ = word_vector_dict[word]
        print("Random Embedding Count: {} \nWord Vector Embedding Count: {}".format(random_count, word_vector_count))


if __name__ == '__main__':
    unittest.main()

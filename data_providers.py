from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec, KeyedVectors
import torch.utils.data as data
from utils import *
import torch
import torch.utils.data
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

    def __init__(self, inputs, targets, seed, transform=None, make_one_hot=False):
        self.inputs = np.array(inputs)
        self.num_classes = len(set(targets))
        self.transform = transform

        if make_one_hot:
            self.targets = self.to_one_of_k(targets)
        else:
            self.targets = np.array(targets)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sample = self.inputs[index]
        if self.transform:
            sample = self.transform(sample)

        return sample, self.targets[index]

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
    def __init__(self, path_data, path_labels, experiment_flag):
        self.experiment_flag = experiment_flag
        label_data = pd.read_csv(path_labels, header='infer', index_col=0, squeeze=True).to_dict()
        data = np.load(os.path.join(ROOT_DIR, path_data))
        data = data[()]
        self.outputs, self.labels = extract_tweets(label_data, data, self.experiment_flag)

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
        elif 'fasttext' in key:
            print("[Model] Using {} embeddings".format(key))
            embed_dim = FASTTEXT_EMBED_DIM
            if key == 'fasttext-wiki':
                filename = os.path.join(ROOT_DIR, 'data/fasttext/wiki-news-300d-1M.vec')
            elif key == 'fasttext-wiki-subword':
                filename = os.path.join(ROOT_DIR, 'data/fasttext/wiki-news-300d-1M-subword.vec')
            elif key == 'fasttext-crawl':
                filename = os.path.join(ROOT_DIR, 'data/fasttext/crawl-300d-2M.vec')
            elif key == 'fasttext-crawl-subword':
                filename = os.path.join(ROOT_DIR, 'data/fasttext/crawl-300d-2M-subword.vec')
            word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        else:
            print("[Model] Using {} embeddings".format(key))
            embed_dim = EMBED_DIM
            model = word2vec.Word2Vec(sentences=tweets_corpus, size=embed_dim)
            model.train(tweets_corpus, total_examples=len(tweets_corpus), epochs=100)
            word_vectors = model.wv
        return word_vectors, embed_dim

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

    @staticmethod
    def fetch_word_embeddings(outputs, word_vectors, embed_dim, experiment_flag=2):
        tweet_sentence_size = TWEET_SENTENCE_SIZE
        if experiment_flag == 2:
            tweet_sentence_size *= 2

        outputs_embed = []
        for i, output in enumerate(outputs):
            tweet = output['tokens']
            embedded_tweet = []

            # trim if too large
            if len(tweet) >= tweet_sentence_size:
                tweet = tweet[:tweet_sentence_size]

            # convert all into word embeddings
            for word in tweet:
                embedding = generate_random_embedding(embed_dim) if word not in word_vectors else word_vectors[word]
                embedded_tweet.append(embedding)

            # pad if too short
            if len(tweet) < tweet_sentence_size:
                diff = tweet_sentence_size - len(tweet)
                embedded_tweet += [generate_random_embedding(embed_dim) for _ in range(diff)]

            assert len(embedded_tweet) == tweet_sentence_size
            output['embedding'] = embedded_tweet
            outputs_embed.append(output)
        return outputs_embed

    def generate_tdidf_embeddings(self, seed):
        x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(self.outputs, self.labels, seed)

        x_train = convert_to_feature_embeddings(x_train, key='tokens')
        x_valid = convert_to_feature_embeddings(x_valid, key='tokens')
        x_test = convert_to_feature_embeddings(x_test, key='tokens')

        vectorizer = TfidfVectorizer(use_idf=True, max_features=10000)

        return {'x_train': vectorizer.fit_transform(x_train).todense(),
                'y_train': y_train,
                'x_valid': vectorizer.transform(x_valid).todense(),
                'y_valid': y_valid,
                'x_test': vectorizer.transform(x_test).todense(),
                'y_test': y_test}

    def generate_word_level_embeddings(self, embedding_key, seed):
        x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(self.outputs, self.labels, seed)
        word_vectors, embed_dim = self._fetch_model(x_train, embedding_key)

        # embed inputs
        x_train_embed = self.fetch_word_embeddings(x_train, word_vectors, embed_dim, self.experiment_flag)
        x_valid_embed = self.fetch_word_embeddings(x_valid, word_vectors, embed_dim, self.experiment_flag)
        x_test_embed = self.fetch_word_embeddings(x_test, word_vectors, embed_dim, self.experiment_flag)

        return {'x_train': convert_to_feature_embeddings(x_train_embed),
                'y_train': y_train,
                'x_valid': convert_to_feature_embeddings(x_valid_embed),
                'y_valid': y_valid,
                'x_test': convert_to_feature_embeddings(x_test_embed),
                'y_test': y_test
                }

    def generate_char_level_embeddings(self, seed):
        raw_tweets = self.tokenize(self.raw_tweets)
        processed_tweets = self.fetch_character_embeddings(raw_tweets)
        return self._generate_embedding_output(processed_tweets, seed)

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec, KeyedVectors
import torch.utils.data as data
from utils import *
import torch
import torch.utils.data
import pandas as pd
from bert_embedding import BertEmbedding

GOOGLE_EMBED_DIM = 300
TWITTER_EMBED_DIM = 400
TWEET_SENTENCE_SIZE = 17  # 16 is average tweet token length
TWEET_WORD_SIZE = 20 # selected by histogram of tweet counts
FASTTEXT_EMBED_DIM = 300
EMBED_DIM = 200
NUM_CLASSES = 4
BERT_EMBEDDING_NUM = 11
TDIDF_MAX_FEATURES = 500
BERT_EMBED_DIM = 768

class DataProvider(data.Dataset):
    """Generic data provider."""

    def __init__(self, inputs, targets, seed):
        self.inputs = list(inputs)
        self.targets = targets
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


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
            inputs, label = dataset[idx][0], dataset[idx][1]
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
    def __init__(self, path_data, path_labels, experiment_flag, embedding_key):
        self.experiment_flag = experiment_flag

        # create ouputs with tweet data
        label_data = pd.read_csv(path_labels, header='infer', index_col=0, squeeze=True).to_dict()
        data = np.load(os.path.join(ROOT_DIR, path_data), allow_pickle=True)
        data = data[()]
        self.outputs, self.labels = extract_tweets(label_data, data, self.experiment_flag)
        self.embedding_key = embedding_key

        # populate outputs with specific embeddings
        if embedding_key == 'twitter':
            self.embed_dim = TWITTER_EMBED_DIM
            filename = os.path.join(ROOT_DIR, 'data/word2vec_twitter_model/word2vec_twitter_model.bin')
            self.word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
            self.generate_twitter_embeddings()
        elif embedding_key == 'bert':
            self.embed_dim = BERT_EMBED_DIM
            self.bert_embeddings = self.generate_bert_embedding_dict()
            self.generate_bert_embeddings()
            self.bert_embedding_generator = BertEmbedding()
        elif embedding_key == 'tdidf':
            self.embed_dim = TDIDF_MAX_FEATURES
            self.vectorizer = None

    @staticmethod
    def generate_bert_embedding_dict():
        embeddings = {}
        for i in range(BERT_EMBEDDING_NUM):
            results = np.load(os.path.join(ROOT_DIR, 'data/bert_embeddings_{}.npz'.format(i)), allow_pickle=True)
            print("Downloading Bert, Processed {} / {}".format(i+1, BERT_EMBEDDING_NUM))
            results = results['a']
            results = results[()]
            embeddings = {**results, **embeddings}
        return embeddings

    @staticmethod
    def process_tweet(tweet, embed_dim, word_vectors):
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
        return embedded_tweet

    def embed_words(self, words, scores=None):
        embedded_words = []
        if self.embedding_key == 'twitter':
            # convert all into word embeddings
            for word in words:
                embedding = generate_random_embedding(self.embed_dim) if word not in self.word_vectors else self.word_vectors[word]
                embedded_words.append(embedding)
        elif self.embedding_key == 'bert':
            items = self.bert_embedding_generator(words)
            for item in items:
                try:
                    embedded_words.append(item[1][0])
                except:
                    embedded_words.append(np.zeros((self.embed_dim,)))
        elif self.embedding_key == 'tdidf':
            embedded_words = np.array(self.vectorizer.transform(words).todense())

        if scores:
            embedded_words = self.add_scores(embedded_words, 10, scores)
        embedded_words = np.array(embedded_words)
        return embedded_words

    @staticmethod
    def add_scores(embeds, word_count, scores):
        features = np.array([scores for _ in range(word_count)])  # adding 1
        embed = np.concatenate((embeds, features), -1)
        return embed

    def generate_twitter_embeddings(self):
        # if self.experiment_flag == 4:
        #     user_lda_scores = np.load(os.path.join(ROOT_DIR, 'data/user_lda_scores_final.npz'))
        #     user_lda_scores = user_lda_scores['a'][()]
        #     print("Length of lda scores {}".format(len(user_lda_scores)))
        for j, (key, output) in enumerate(self.outputs.items()):

            # process first tweet
            embedded_tweet = self.process_tweet(output['tokens'], self.embed_dim, self.word_vectors)
            assert len(embedded_tweet) == TWEET_SENTENCE_SIZE
            self.outputs[key]['embedded_tweet'] = embedded_tweet

            if self.experiment_flag == 2:
                embedded_context_tweet = []
                if output['context_tweet'] is None:
                    for _ in range(TWEET_SENTENCE_SIZE):
                        blank_embedding = np.zeros(self.embed_dim,)
                        embedded_context_tweet.append(blank_embedding)
                else:
                    context_embedding = self.process_tweet(output['context_tokens'], self.embed_dim, self.word_vectors)
                    for j in range(TWEET_SENTENCE_SIZE):
                        embedded_context_tweet.append(context_embedding[j])
                assert len(embedded_context_tweet) == TWEET_SENTENCE_SIZE
                self.outputs[key]['embedded_context_tweet'] = embedded_context_tweet

            elif self.experiment_flag == 5:
                if j % 1000 == 0:
                    print("Processing tweet {}/{}".format(j, len(self.outputs)))
                user_timeline = self.outputs[key]['user_timeline']
                user_timeline_processed = [self.process_tweet(tweet, self.embed_dim, self.word_vectors)
                                           for tweet in user_timeline]
                output[key]['embedded_user_timeline'] = user_timeline_processed

    def generate_tdidf_embeddings(self, seed):
        x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(list(self.outputs.keys()), self.labels, seed)
        self.vectorizer = TfidfVectorizer(use_idf=True, max_features=TDIDF_MAX_FEATURES)

        for i, _set in enumerate([x_train, x_valid, x_test]):
            if i == 0:
                self.vectorizer.fit([self.outputs[key]['tweet'] for key in _set])

            embedded_tweets = self.vectorizer.transform([self.outputs[key]['tweet'] for key in _set]).todense()
            embedded_context_tweets = self.vectorizer.transform(
                [self.outputs[key]['context_tweet'] if self.outputs[key]['context_tweet'] is not None else '' for key in
                 _set]).todense()
            embedded_topics = self.vectorizer.transform(
                [' '.join(self.outputs[key]['user_topic_tokens']) for key in _set]).todense()

            perplexity_mean = np.mean([self.outputs[key]['perplexity'] for key in _set])
            cohesion_mean = np.mean([self.outputs[key]['cohesion'] for key in _set])
            features = np.array([[perplexity_mean, cohesion_mean] for _ in range(len(_set))])
            embedded_topics = np.concatenate((np.array(embedded_topics), features), -1)

            # if self.experiment_flag == 5:
            #     pass
            #     # embedded_timeline_tweets = []
            #     # for i in range(200):
                #     for key in _set:
                #     self.outputs[key]['embedded_tweet_user_timeline_{}'.format(i)] = vectorizer.transform(
                #         [self.outputs[key]['user_timeline_tokens_{}'].format(i) for key in _set
                #          if 'user_timeline_tokens_{}'.format(i) in self.outputs[key]]).todense())

            for i, key in enumerate(_set):
                self.outputs[key]['embedded_tweet'] = np.array(embedded_tweets[i])
                self.outputs[key]['embedded_context_tweet'] = np.array(embedded_context_tweets[i])
                self.outputs[key]['embedded_topic_words'] = np.array(embedded_topics[i])

        return {'x_train': x_train,
                'y_train': y_train,
                'x_valid': x_valid,
                'y_valid': y_valid,
                'x_test': x_test,
                'y_test': y_test}, self.outputs

    def generate_bert_embeddings(self):
        """
        :param embeddings: preprocessed bert word embeddings
        :param data: has all fields, separate fn from gen_word_embeddings
        :return:
        """
        for key, output in self.outputs.items():
            self.outputs[key]['embedded_tweet'] = self.bert_embeddings[int(output['id'])]
        if self.experiment_flag == 2 or self.experiment_flag == 4:
            bert_embedded_topic_words = aggregate(0, 65, 'bert/bert_topic_words')

            # finding bert embedding for reply tweet
            for i, (key, output) in enumerate(self.outputs.items()):
                if self.experiment_flag == 2:
                    tweet_embed = self.bert_embeddings[int(output['id'])]
                    self.outputs[key]['embedded_tweet'] = tweet_embed

                    reply_status_id = int(output['in_reply_to_status_id'])
                    blank_embed = []
                    if reply_status_id == -1 or reply_status_id not in self.bert_embeddings:
                        for i in range(17):
                            blank_embedding = np.zeros(BERT_EMBED_DIM, )
                            blank_embed.append(blank_embedding)
                        embedded_context_tweet = blank_embed
                    else:
                        embedded_context_tweet = self.bert_embeddings[reply_status_id]
                    self.outputs[key]['embedded_context_tweet'] = embedded_context_tweet
                if self.experiment_flag == 4:
                    if output['user_id'] in bert_embedded_topic_words:
                        embedded_topic_words = bert_embedded_topic_words[output['user_id']]
                        print(embedded_topic_words.shape)
                        self.outputs[key]['embedded_topic_words'] = embedded_topic_words
                    else:
                        self.outputs[key]['embedded_topic_words'] = np.zeros((10, BERT_EMBED_DIM + 2))

    def generate_word_level_embeddings(self, seed):
        x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(list(self.outputs.keys()), self.labels, seed)
        print("Word embeddings generated")
        return {'x_train': x_train,
                'y_train': y_train,
                'x_valid': x_valid,
                'y_valid': y_valid,
                'x_test': x_test,
                'y_test': y_test
                }, self.outputs

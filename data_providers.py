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
    def __init__(self, path_data, path_labels, experiment_flag, bert_embeddings = None):
        self.bert_embeddings = bert_embeddings
        self.experiment_flag = experiment_flag
        label_data = pd.read_csv(path_labels, header='infer', index_col=0, squeeze=True).to_dict()
        data = np.load(os.path.join(ROOT_DIR, path_data), allow_pickle=True)
        data = data[()]
        self.outputs, self.labels = extract_tweets(label_data, data, self.experiment_flag)

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

    @staticmethod
    def twitter_embed_words(words, embed_dim, word_vectors):
        embedded_words = []
        # convert all into word embeddings
        for word in words:
            embedding = generate_random_embedding(embed_dim) if word not in word_vectors else word_vectors[word]
            embedded_words.append(embedding)
        return np.array(embedded_words)

    def generate_twitter_embeddings(self):
        embed_dim = TWITTER_EMBED_DIM
        filename = os.path.join(ROOT_DIR, 'data/word2vec_twitter_model/word2vec_twitter_model.bin')
        word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
        user_lda_scores = np.load(os.path.join(ROOT_DIR, 'data/user_lda_scores_final.npz'))
        user_lda_scores = user_lda_scores['a'][()]
        print("Length of lda scores {}".format(len(user_lda_scores)))
        for key, output in self.outputs.items():

            # process first tweet
            embedded_tweet = self.process_tweet(output['tokens'], embed_dim, word_vectors)
            assert len(embedded_tweet) == TWEET_SENTENCE_SIZE
            self.outputs[key]['embedded_tweet'] = embedded_tweet

            if self.experiment_flag == 2:
                embedded_context_tweet = []
                if output['context_tweet'] is None:
                    for _ in range(TWEET_SENTENCE_SIZE):
                        blank_embedding = np.zeros(embed_dim,)
                        embedded_context_tweet.append(blank_embedding)
                else:
                    context_embedding = self.process_tweet(output['context_tokens'], embed_dim, word_vectors)
                    for j in range(TWEET_SENTENCE_SIZE):
                        embedded_context_tweet.append(context_embedding[j])
                assert len(embedded_context_tweet) == TWEET_SENTENCE_SIZE
                self.outputs[key]['embedded_context_tweet'] = embedded_context_tweet

            elif self.experiment_flag == 4:
                if output['user_id'] in user_lda_scores:
                    perplexity, cohesion, topic_words = user_lda_scores[output['user_id']]
                    topic_words = [word for (word, _) in topic_words]
                    if len(topic_words) < 10:
                        for _ in range(10 - len(topic_words)):
                            topic_words.append(' ')
                    embedded_topic_words = self.twitter_embed_words(topic_words, embed_dim, word_vectors)
                    embedded_topic_words = self.add_scores(embedded_topic_words,
                                                           len(embedded_topic_words),
                                                           perplexity,
                                                           cohesion)
                    self.outputs[key]['embedded_topic_words'] = embedded_topic_words
                    self.outputs[key]['embedded_tweet_perplexity_cohesion'] = self.add_scores(embedded_tweet,
                                                                                              len(embedded_tweet),
                                                                                              perplexity,
                                                                                              cohesion)
                else:
                    self.outputs[key]['embedded_topic_words'] = np.zeros((10, embed_dim + 2))
                    self.outputs[key]['embedded_tweet_perplexity_cohesion'] = self.add_scores(embedded_tweet,
                                                                                              len(embedded_tweet),
                                                                                              0,
                                                                                              0)

    def generate_tdidf_embeddings(self, seed):
        x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(list(self.outputs.keys()), self.labels, seed)
        vectorizer = TfidfVectorizer(use_idf=True, max_features=TDIDF_MAX_FEATURES)

        for _set in [x_train, x_valid, x_test]:
            embedded_tweets = vectorizer.fit_transform([self.outputs[key]['tweet'] for key in _set]).todense()
            embedded_context_tweets = vectorizer.fit_transform(
                [self.outputs[key]['context_tweet'] if self.outputs[key]['context_tweet'] is not None else '' for key in
                 _set]).todense()
            for i, key in enumerate(_set):
                self.outputs[key]['embedded_tweet'] = np.array(embedded_tweets[i])
                self.outputs[key]['embedded_context_tweet'] = np.array(embedded_context_tweets[i])

        return {'x_train': x_train,
                'y_train': y_train,
                'x_valid': x_valid,
                'y_valid': y_valid,
                'x_test': x_test,
                'y_test': y_test}, self.outputs

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
    def add_scores(embeds, word_count, perplexity, cohesion):
        features = np.array([[perplexity, cohesion] for _ in range(word_count)])  # addding 1
        embed = np.concatenate((embeds, features), -1)
        return embed

    @staticmethod
    def bert_embed_words(words, bert_embedding):
        result = []
        items = bert_embedding(words)
        for item in items:
            try:
                result.append(item[1][0])
            except:
                result.append(np.zeros((768,)))
        return result

    def generate_bert_embeddings(self, outputs):
        """
        :param embeddings: preprocessed bert word embeddings
        :param data: has all fields, separate fn from gen_word_embeddings
        :return:
        """
        user_lda_scores = aggregate(22, 'user_lda_scores')
        bert_embedding = BertEmbedding()

        if self.experiment_flag == 1 or self.experiment_flag == 3:
            for key, output in outputs.items():
                outputs[key]['embedded_tweet'] = self.bert_embeddings[int(output['id'])]
        else:
            # finding bert embedding for reply tweet
            for i, (key, output) in enumerate(outputs.items()):
                if self.experiment_flag == 2:
                    tweet_embed = self.bert_embeddings[int(output['id'])]
                    outputs[key]['embedded_tweet'] = tweet_embed

                    reply_status_id = int(output['in_reply_to_status_id'])
                    blank_embed = []
                    if reply_status_id == -1 or reply_status_id not in self.bert_embeddings:
                        for i in range(17):
                            blank_embedding = np.zeros(BERT_EMBED_DIM, )
                            blank_embed.append(blank_embedding)
                        embedded_context_tweet = blank_embed
                    else:
                        embedded_context_tweet = self.bert_embeddings[reply_status_id]
                    outputs[key]['embedded_context_tweet'] = embedded_context_tweet
                if self.experiment_flag == 4:
                    print("{}".format(i))
                    if output['user_id'] in user_lda_scores:
                        perplexity, cohesion, topic_words = user_lda_scores[output['user_id']]
                        topic_words = [word for (word, _) in topic_words]
                        if len(topic_words) < 10:
                            for _ in range(10 - len(topic_words)):
                                topic_words.append(' ')
                        embedded_topic_words = self.bert_embed_words(topic_words, bert_embedding)
                        embedded_topic_words = self.add_scores(embedded_topic_words, len(embedded_topic_words))
                        self.outputs[key]['embedded_topic_words'] = embedded_topic_words
                    else:
                        self.outputs[key]['embedded_topic_words'] = np.zeros((10, BERT_EMBED_DIM + 2))

    def generate_word_level_embeddings(self, embedding_key, seed):
        x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(list(self.outputs.keys()), self.labels, seed)

        if embedding_key == 'bert':
            if self.bert_embeddings is None:
                self.bert_embeddings = self.generate_bert_embedding_dict()
            self.generate_bert_embeddings(self.outputs)
        else:
            self.generate_twitter_embeddings()

        print("Word embeddings generated")
        return {'x_train': x_train,
                'y_train': y_train,
                'x_valid': x_valid,
                'y_valid': y_valid,
                'x_test': x_test,
                'y_test': y_test
                }, self.outputs

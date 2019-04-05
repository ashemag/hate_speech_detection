"""
Source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
"""
import numpy as np
import pickle
import bcolz

OUTPUT_PATH = 'data/glove.6B/pickle/'
EMBED_DIM = 50


class Glove():
    def __init__(self):
        self.glove = None
        self.vocabulary = None
        self.weights_matrix = None

    @staticmethod
    def glove_load(filename):
        """
        NEED ONLY BE CALLED ONCE
        Used to initially load in GloVe data into pickled data structure
        :param filename: filename of GloVe .txt file
        :return:
        """
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=OUTPUT_PATH + '6B.50.dat', mode='w')

        with open(filename, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)

        vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=OUTPUT_PATH + '6B.50.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open(OUTPUT_PATH + '6B.50_words.pkl', 'wb'))
        pickle.dump(word2idx, open(OUTPUT_PATH + '6B.50_idx.pkl', 'wb'))

    def glove_fetch(self):
        """
        Fetches glove data structure from stored pickled object
        :return:
        """
        print("Accessing pickled glove data...")
        vectors = bcolz.open(OUTPUT_PATH + '6B.50.dat')[:]
        words = pickle.load(open(OUTPUT_PATH + '6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(OUTPUT_PATH + '6B.50_idx.pkl', 'rb'))
        print("Populating dict...")
        glove = {w: vectors[word2idx[w]] for w in words}
        self.glove = glove

    def get_vocabulary(self, x_train):
        """
        Gets vocabulary from training set of tweets
        :param x_train: training set of tweet objects
        :return:
        """
        vocabulary = set()
        for words in x_train:
            vocabulary |= set(words)
        self.vocabulary = vocabulary

    def generate_weight_matrix(self, vocabulary=None):
        """
        For each word in datasets vocabulary, we check if its in GloVe
        if yes, we load pretrained word vector
        if not we initialize random vector
        :param vocabulary: vocabulary of tweets in training set
        :return:
        """
        if vocabulary is None:
            if self.vocabulary is None:
                raise ValueError("Please extract vocabulary from training set.")
            else:
                vocabulary = self.vocabulary

        matrix_len = len(vocabulary)
        weights_matrix = np.zeros((matrix_len, EMBED_DIM))
        words_found = 0

        for i, word in enumerate(vocabulary):
            try:
                weights_matrix[i] = self.glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBED_DIM,))

        self.weights_matrix = weights_matrix

    def create_emb_layer(self, non_trainable=False):
        """
        To create an embedding layer for Neural Net
        :param non_trainable:
        :return:
        """
        num_embeddings, embedding_dim = self.weights_matrix.shape
        emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': torch.Tensor(self.weights_matrix)})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

# TESTING
g = Glove()
g.glove_fetch()
vocab = {'', 'retarded', 'prime', 'nancy', 'gullible', 'weave', 'skittles', 'niggas', 'asians', 'awakens', 'gonna', 'yall', 'hotmama', 'pray', 'always', 'traded', 'birds', 'butt', 'ma', 'dick', 'bronco', 'really', 'hell', 'school', 'dare', 'lmfao', 'dummy', 'females', 'salty', 'baekhyuns', 'insulting', 'police', 'ugly', 'hamill', 'slip', 'difference', 'sid', 'production', 'shits', 'want', 'sick', 'carlos', 'access', 'history', 'outta', 'asking', 'vote', 'moron', 'send', 'reason', 'hands', 'wrestlemania', 'walkup', 'also', 'hoe', '\n\npull', 'ask', 'civil', 'money', 'music', 'meets', 'ig', 'stuck', 'asshole', 'cruiserweight', 'days', 'deeply', 'bag', 'sisters', 'systems', 'fools', 'ugh', 'notmypresident', 'pizza', 'still', 'hemet', 'policy\nrussia', 'bruh', 'going', 'gas', 'people', 'painful', 'youre', 'officer', 'country', 'shipping', 'smoking', 'combo', 'impeachtrump', 'evil', 'started', 'sdlive', 'maga', 'kids', 'miss', 'dope', 'jesus', 'draintheswamp', 'rt', 'working', 'god', 'talk', 'bc', 'go', 'texas', 'saying', 'happening', 'show', 'literally', 'force', 'stickers', 'reading', 'id', 'take', 'done', 'pov', 'afterburners', 'ooc', 'sampled', 'ass', 'stream', 'praise', 'fight', 'gotta', 'every', '\n\ntheresistance', 'looking', 'moving', 'fblchat', 'messages', 'dur', 'bite', 'grassley', 'watching', 'seeing', 'voted', 'unless', 'fuck', 'something', 'reagan', 'someone', 'dicks', 'buying', 'consequences', 'degrading', 'fro', 'freaking', 'han', 'around', 'feared', 'person', 'yes', 'sis', 'another', 'happens', 'aint', 'tell', 'yk', 'scare', 'swear', 'mortis', 'pissing', 'parks', 'code\nlet', 'er', 'pistol', 'knows', 'chase', 'care', 'sloth', 'lying', 'change', 'charity', 'handling', 'sake', 'freak', 'mud', 'stillwithher', 'oral', 'got', 'mkr', 'pis', 'papaya', 'boombap', 'tired', 'business', 'race', 'cheated', 'see', 'opened', 'poems', 'lesson', 'probably', 'telling', 'uniform', 'flock', 'give', 'post', 'stupid', 'guys', 'bout', 'aldubksgoestous', 'dumbass', 'tru', 'trust', 'ur', 'backyard', 'write', 'omg', 'told', 'fact', 'chanyeol', 'lie', 'sleep', 'lip', 'thats', 'duncan', 'thinks', 'forgot', 'velvet', 'nigga', 'dinny', 'yooooo', 'nuclearo', 'sounds', 'fucking', 'work', 'twittertumblr', 'solo', 'democracy', 'dam', 'regime', 'spring', 'replies', 'sit', 'disneyland\n\nsomeone', 'bcs', 'worst', 'brosas', 'dawg', 'mad', 'trumprussiaco', 'tea', 'u', 'court', 'laughing', 'worried', 'bird', 'real', 'rocking', 'appropo', 'owlshow', 'dad', 'watched', 'address', 'heres', 'fan', 'drinking', 'belong', 'nail', 'party', 'biggest', 'break', 'using', 'bitches', 'flame', 'tonight', 'allow', 'guy', 'ago', 'hear', 'death', 'featuring', 'repubs', 'made', 'must', 'min', 'turd', 'dean', 'walk', 'holds', 'hmu', 'yang', 'disini', 'leaked', 'think', 'cry', 'like', 'friend', 'hoes', 'mine', 'susah', 'actually', 'spend', 'vans', 'ol', 'wanna', 'oda', 'girl', 'antacids', 'boring', 'abc', 'marketing', 'anon', 'gyalchester', 'match', 'kidding', 'pairings', 'one', 'assclown', 'correa', 'amp', 'didnt', 'get', 'gone', 'city', 'die', 'sad', 'nasty', 'fruit', 'crap', 'know', 'seen', 'splicers', 'fbloggers', 'everything', 'disrespects', 'hate', 'whine', 'ship', 'pathetic', 'ged', 'consu', 'put', 'look', 'repeatedly', 'went', 'goldenera', 'supporters', 'fucks', 'rice', 'damn', 'learn', 'lmfaoooooooooooooooo', 'holy', 'card', 'hurt\n\n', 'repeat', 'reboot\nriffotronic', 'pushing', 'cant', 'goosebumps', 'trump', 'shes', 'sdliveaftermania', 'theres', 'women', 'secretly', 'somebody', 'letang', 'bein', 'anoda', 'backflips', 'idiots', 'wearing', 'tracy', 'tried', 'terrifying', 'feel', 'miserable', 'home', 'oh', 'weytin', 'coming', 'beef', 'boii', 'corbin', 'day', 'april', 'would', 'disgusting', 'htt', 'mark', 'alex', 'mom', 'f', 'fucked', 'no\nor', 'feeling', 'nani', 'never', 'sasha', 'nazi', 'ignore', 'eat', 'rigor', 'gaslight', 'd', 'wrong', 'ya', 'pre', 'say', 'annoying', 'negan', 'jaws', 'speak', 'flagrantly', 'pencil', 'reply', 'bad', 'awful', 'onnnnssssss', 'hold', 'mayor', 'braids', 'hole', 'wash', 'call', 'talking', 'ever', 'health', 'internet', 'news', 'gon', 'children', 'works', 'stay', 'itu', 'longmelbourne', 'showed', 'away', 'unfollowing', 'idiot', 'oxblood', 'fuckkkk', 'sisim', 'gruesome', 'goddamn', 'even', 'built', 'hitler', 'faster', 'hes', 'make', 'doesnt', 'dont', 'case', 'wellwhinesnifflecoughcry', 'pic', 'wanted', 'push', 'mind', 'email', 'pineapples', 'beer', 'wallet', 'team', 'johnny', 'horrific', 'joke', 'bring', 'budget', 'btw', 'office', 'keep', 'follow', 'liar', 'untouchable', 'tweet', 'trying', 'called', 'find', 'words', 'almost', 'bros', 'dreadful', 'dadahati', 'way', 'everyone', 'pulling', 'looks', 'via', 'titts', 'dog', 'republican', 'sequel', 'extended', 'bloody', 'dr', 'fanbase', 'sacrifice', 'says', 'litter', 'pathological', 'terrible\nyou', 'im', 'hand', 'face', 'sealed', 'hiphop', 'salt', 'fuckin', 'tumblr', 'sugar', 'flynn', 'suck', 'bitch', 'million', 'kill', 'times', 'mak', 'interview', 'sum', 'yo', 'jiggy', 'rewarded'}
g.generate_weight_matrix(vocab)
print(g.create_emb_layer())
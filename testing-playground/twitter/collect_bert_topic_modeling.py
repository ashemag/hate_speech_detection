from bert_embedding import BertEmbedding
import torch
import sys
import os
import numpy as np
import pandas as pd
sys.path.append("..")
from utils import *
from globals import ROOT_DIR


def bert_embed_words(words, bert_embedding):
    result = []
    items = bert_embedding(words)
    for item in items:
        try:
            result.append(item[1][0])
        except:
            result.append(np.zeros((768,)))
    return result


def add_scores(embeds, word_count):
    features = np.array([[1, 1] for _ in range(word_count)]) #addding 1
    embed = np.concatenate((embeds, features), -1)
    return embed

# for all tweets in our data, compute topic embeddings
embedded_topic_inputs = {}
bert_embedding = BertEmbedding()

#aggregate user ldad scores
def aggregate(start, end, file_names):
    aggregate_data = {}
    for i in range(start, end):
        path_name = os.path.join(ROOT_DIR, 'data/{}_{}.npz'.format(file_names, i))
        results = np.load(path_name, allow_pickle=True)
        print("Downloading {}, Processed {} / {}".format(path_name, i+1, end))
        results = results['a']
        results = results[()]
        print(len(results))
        aggregate_data = {**results, **aggregate_data}
    return aggregate_data

timelines = aggregate(0, 66, 'user_timeline')
print(len(timelines))
exit()
# dataset
data_file = os.path.join(ROOT_DIR, 'data/founta_data.npy')
data = np.load(data_file)
data = data[()]
path_labels = os.path.join(ROOT_DIR, 'data/founta_data.csv')
labels = pd.read_csv(path_labels, header='infer', index_col=0, squeeze=True).to_dict()


path_name = os.path.join(ROOT_DIR, 'data/{}.npz'.format('user_lda_scores_main'))
user_lda_scores = np.load(path_name, allow_pickle=True)
user_lda_scores = user_lda_scores['a']
user_lda_scores = user_lda_scores[()]
import time
start = time.time()
save_count = 0

for i, (key, value) in enumerate(data.items()):
    user_id = value['user']['id']
    if user_id in user_lda_scores:
        # retrieve word embeddings
        perplexity, cohesion, topic_words = user_lda_scores[user_id]
        topic_words = [word for (word, _) in topic_words]
        if len(topic_words) < 10:
            for _ in range(10 - len(topic_words)):
                topic_words.append(' ')
        embedded_topic_words = bert_embed_words(topic_words, bert_embedding)
        embedded_topic_words = add_scores(embedded_topic_words, len(embedded_topic_words))
        embedded_topic_inputs[value['id']] = embedded_topic_words
    else:
        embedded_topic_inputs[value['id']] = np.zeros((10, 768+2)) #0 embedded if not found
    if i % 100 == 0:
        print("On {}, {:.2f} min".format(i, ((time.time()) - start) / 60))
    if i % 1000 == 0:
        np.savez(os.path.join(ROOT_DIR, 'data/bert_topic_words_{}.npz'.format(save_count)), a=embedded_topic_inputs)
        embedded_topic_inputs = {}
        save_count += 1 # twitter id -> bert topic words
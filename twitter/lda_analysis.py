import sys
import os
sys.path.append("..")
from globals import ROOT_DIR
import configparser
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import numpy as np
from collections import Counter


timelines = {}
# process 2 and 3
# 0, 1, 2, 3
for i in [3]:
    results = np.load(os.path.join(ROOT_DIR, 'data/user_timeline_processed_{}.npz'.format(i)), allow_pickle=True)
    print("Downloading User Timelines, Processed {} / {}".format(i+1, 6))
    results = results['a']
    results = results[()]
    timelines = {**results, **timelines}

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def get_scores(docs, nlp):
    try:
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(docs, nlp, allowed_postags=['NOUN', 'ADJ', 'ADV'])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=5,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=1000,
                                                    passes=1,
                                                    alpha='auto',
                                                    per_word_topics=True)
        # # Print the Keyword in the 10 topics
        dominant_keywords = []
        for i, row_list in enumerate(lda_model[corpus]):
            row = row_list[0] if lda_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = lda_model.show_topic(topic_num)
                    dominant_keywords.extend([word for word, prop in wp])
                    # topic_keywords = ", ".join([word for word, prop in wp])
                else:
                    break
        topic_words = Counter(dominant_keywords).most_common(10)
        # topic_words = []
        # len(doc_lda)
        # Compute Perplexity
        # print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        # print('Coherence Score: ', coherence_lda)
        # In my experience, topic coherence score, in particular, has been more helpful.
        #vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

        return lda_model.log_perplexity(corpus), coherence_lda, topic_words
    except:
        return None, None, None

nlp = spacy.load('en', disable=['parser', 'ner'])
user_scores = {}
save_count = 20
import time
start = time.time()
for i, (key, value) in enumerate(timelines.items()):
    if i % 100 == 0:
        print("{}) {} min".format(i, (time.time() - start)/60))

    value = value[:200]
    doc = [tweet for tweet in value]
    if len(doc) == 0:
        continue
    perplexity, coherence, topic_words = get_scores(doc, nlp)
    if perplexity is None:
        continue
    user_scores[key] = [perplexity, coherence, topic_words]

    if i % 1000 == 0 and i != 0:
        np.savez(os.path.join(ROOT_DIR, 'data/user_lda_scores_{}.npz'.format(save_count)), a=user_scores)
        save_count += 1
        user_scores = {}
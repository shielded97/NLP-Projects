# run statistics on the trained LDA model
# KLDiv implementation was done through the tutorial found here: 
# https://radimrehurek.com/gensim_3.8.3/auto_examples/tutorials/run_distance_metrics.html

from gensim import models
from gensim.models import ldamodel
from gensim.models.ldamodel import LdaModel
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import gensim
import io
import os
import re
import pandas as pd
import numpy as np
import logging
import pprintpp
import pyLDAvis
from pyLDAvis import gensim_models
from gensim.matutils import kullback_leibler

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None 

def remove_stop(df):
    for review in df:
        yield(gensim.parsing.preprocessing.remove_stopwords(str(review)))

def tag_converter(tag):
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:          
        return None

reviews = pd.read_csv('./Review.csv', index_col=0)

reviews = reviews.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', \
    'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Num_words_text'], axis=1)

tokenizer = RegexpTokenizer(r'\w+')
reviews = reviews.values.tolist()
for idx in range(len(reviews)):
    reviews[idx] = str(reviews[idx]).lower()  
    reviews[idx] = tokenizer.tokenize(reviews[idx])
    reviews[idx] = nltk.pos_tag(reviews[idx])
print('Review 0: ', reviews[0])
lemmatizer = nltk.stem.WordNetLemmatizer()
for idx in range(len(reviews)):
    wordnet_tagged = map(lambda x: (x[0], tag_converter(x[1])), reviews[idx])
    reviews[idx] = []
    for word, tag in wordnet_tagged:
        if tag is None:
            reviews[idx].append(lemmatizer.lemmatize(word))
        else:
            reviews[idx].append(lemmatizer.lemmatize(word, tag))

print('Lemmatized Review 0: ', reviews[0])

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = models.Phrases(reviews, min_count=5)
for idx in range(len(reviews)):
    for token in bigram[reviews[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            reviews[idx].append(token)

dictionary = gensim.corpora.Dictionary(reviews)
dictionary.filter_extremes(no_below=10, no_above=0.5)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(review) for review in reviews]

model = LdaModel.load('models/big/LDA_Alp0_Eta2.model')
model.show_topics()

rev1 = dictionary.doc2bow(reviews[0])
rev2 = dictionary.doc2bow(reviews[1])
rev3 = dictionary.doc2bow(reviews[2])

lda_bow_rev1 = model.get_document_topics(rev1, minimum_probability=0.0)
lda_bow_rev2 = model.get_document_topics(rev2, minimum_probability=0.0)
lda_bow_rev3 = model.get_document_topics(rev3, minimum_probability=0.0)

topic2 = model.show_topic(0, topn=30)
topic3 = model.show_topic(3, topn=30)

topic2word = []
topic3word = []

for word, num in topic2:
    topic2word.append(word)

for word, num in topic3:
    topic3word.append(word)

topic2bow = dictionary.doc2bow(topic2word)
topic3bow = dictionary.doc2bow(topic3word)

topic2lda = model.get_document_topics(topic2bow, minimum_probability=0.0)
topic3lda = model.get_document_topics(topic3bow, minimum_probability=0.0)

print(kullback_leibler(topic2lda, topic3lda))
print(kullback_leibler(topic3lda, topic2lda))
# run statistics on the trained LDA model


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

f = open("modelCoherenceF.txt", "w")

# calculate topic coherences for each model to evaluate best number of topics, alpha, and eta
num_top = [5, 10, 15, 25, 50]
for idx in range(0,5):
    num_t = 'models/num_topics/LDA_vis_Num_Topics_' + str(idx) + '.model'
    model_n = LdaModel.load(num_t)

    top_topics = model_n.top_topics(corpus, topn=30)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_top[idx]
    f.write(str(num_t) + ': \n')
    f.write('Average topic coherence: %.4f.\n' % avg_topic_coherence)

for idx in range(0,5):
    alp = 'models/alphas/LDA_vis_Alp_' + str(idx) + '.model'
    model_alp = LdaModel.load(alp)

    top_topics = model_alp.top_topics(corpus, topn=30)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    f.write(str(alp) + ': \n')
    f.write('Average topic coherence: %.4f.\n' % avg_topic_coherence)

for idx in range(0,5):
    eta = 'models/etas/LDA_vis_Eta_' + str(idx) + '.model'
    model_eta = LdaModel.load(eta)

    top_topics = model_eta.top_topics(corpus, topn=30)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    f.write(str(eta) + ': \n')
    f.write('Average topic coherence: %.4f.\n' % avg_topic_coherence)

model = LdaModel.load('models/Optimal.model')
top_topics = model_eta.top_topics(corpus, topn=30)
# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / 5
f.write('Optimal Model: \n')
f.write('Average topic coherence: %.4f.\n' % avg_topic_coherence)

f.close()

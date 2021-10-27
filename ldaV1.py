# An LDA implementation that models the topics found inside a dataset of reviews along with data representation
# Much of the code was implemented straight from the LDA tutorial on Gensim's website here: 
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
# Included in that link is the coherence metric that is run to help determine the optimal model parameters
# Lemmatization implementation was learned from here: 
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/


from gensim import models
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

# removes stopwords
def remove_stop(df):
    for review in df:
        yield(gensim.parsing.preprocessing.remove_stopwords(str(review)))

# converts tags into usable tags for nltk lemmatization
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

# read in the reviews
reviews = pd.read_csv('./Review.csv', index_col=0)

# drop unhelpful columns
reviews = reviews.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', \
    'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Num_words_text'], axis=1)

# tokenize the reviews into lists of strings
tokenizer = RegexpTokenizer(r'\w+')
reviews = reviews.values.tolist()
for idx in range(len(reviews)):
    reviews[idx] = str(reviews[idx]).lower()  
    reviews[idx] = tokenizer.tokenize(reviews[idx])
    reviews[idx] = nltk.pos_tag(reviews[idx])

# print('Review 0: ', reviews[0])

# lemmatize words
lemmatizer = nltk.stem.WordNetLemmatizer()
for idx in range(len(reviews)):
    wordnet_tagged = map(lambda x: (x[0], tag_converter(x[1])), reviews[idx])
    reviews[idx] = []
    for word, tag in wordnet_tagged:
        if tag is None:
            reviews[idx].append(lemmatizer.lemmatize(word))
        else:
            reviews[idx].append(lemmatizer.lemmatize(word, tag))

# print('Lemmatized Review 0: ', reviews[0])

# Add bigrams to docs (only ones that appear 5 times or more).
bigram = models.Phrases(reviews, min_count=5)
for idx in range(len(reviews)):
    for token in bigram[reviews[idx]]:
        if '_' in token:
            reviews[idx].append(token)

# build dictionary
dictionary = gensim.corpora.Dictionary(reviews)
dictionary.filter_extremes(no_below=10, no_above=0.5)

# convert to bag of words
corpus = [dictionary.doc2bow(review) for review in reviews]

# print('Number of unique tokens: %d' % len(dictionary))
# print('Number of documents: %d' % len(corpus))

# Now to train the model!!
# Set training parameters
num_topics = [5, 10, 15, 25, 50]
alpha = [0.05, 0.10, 0.15, 0.20, 0.25]
eta = [0.05, 0.10, 0.15, 0.20, 0.25]
chunksize = 2000
passes = 20
iterations = 400
eval_every = None

temp = dictionary[0]
id2word = dictionary.id2token

# open file to write stats to
f = open("modelCoherenceF.txt", "w")

# train different models
for idx in range(len(alpha)):
    for jdx in range(len(eta)):
        model = models.LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha=alpha[idx],
            eta=eta[jdx],
            iterations=iterations,
            num_topics=5,
            passes=passes,
            eval_every=eval_every
        )
        # make visualization
        visualisation = gensim_models.prepare(model, corpus, dictionary)
        name = 'LDA_Alp' + str(idx) + '_Eta' + str(jdx) + '.model'
        vis_name = 'models/big/LDA_Alp' + str(idx) + '_Eta' + str(jdx) + '.html'
        dir = 'models/big/' + name
        model.save(dir)
        pyLDAvis.save_html(visualisation, vis_name)
        top_topics = model.top_topics(corpus, topn=30)
        # compute average topic coherence
        avg_topic_coherence = sum([t[1] for t in top_topics]) / 5
        f.write('Alpha: ' + str(idx) + ' Eta: ' + str(jdx))
        f.write('Average topic coherence: %.4f.\n\n' % avg_topic_coherence)

f.close()
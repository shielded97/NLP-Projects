# from nltk.corpus.reader.wordnet import ADJ, ADV, NOUN, VERB
# from ldaV1 import lemmatize
# import nltk
# nltk.download('wordnet')
# import pandas as pd

# reviews = pd.read_csv('./Review.csv', index_col=0)
# reviews = reviews.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', \
#     'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Num_words_text'], axis=1)

# def lemmatize(df):
#     lemmatizer = nltk.stem.WordNetLemmatizer()
#     for i, review in enumerate(df):
#         if i > 1:
#             break
#         words = nltk.word_tokenize(review)
#         yield(' '.join([lemmatizer.lemmatize(w) for w in words]))

# lemmatizer = nltk.stem.WordNetLemmatizer()

# print(lemmatizer.lemmatize('flavors', pos=NOUN))
# print(lemmatizer.lemmatize('flavors', pos=VERB))
# print(lemmatizer.lemmatize('flavors', pos=ADJ))
# print(lemmatizer.lemmatize('flavors', pos=ADV))

# #print(list(lemmatize(reviews)))


from gensim import models
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
import tarfile
import pandas as pd
import numpy as np
import logging
import pprintpp
import pyLDAvis
from pyLDAvis import gensim_models

from gensim.corpora import Dictionary

# you can use any corpus, this is just illustratory
texts = [
    ['bank','river','shore','water'],
    ['river','water','flow','fast','tree'],
    ['bank','water','fall','flow'],
    ['bank','bank','water','rain','river'],
    ['river','water','mud','tree'],
    ['money','transaction','bank','finance'],
    ['bank','borrow','money'],
    ['bank','finance'],
    ['finance','money','sell','bank'],
    ['borrow','sell'],
    ['bank','loan','sell'],
]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

import numpy
numpy.random.seed(1) # setting random seed to get the same results each time.

from gensim.models import ldamodel
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2, minimum_probability=1e-8)
model.show_topics()
doc_water = ['river', 'water', 'shore']
doc_finance = ['finance', 'money', 'sell']
doc_bank = ['finance', 'bank', 'tree', 'water']

# now let's make these into a bag of words format
bow_water = model.id2word.doc2bow(doc_water)
bow_finance = model.id2word.doc2bow(doc_finance)
bow_bank = model.id2word.doc2bow(doc_bank)

# we can now get the LDA topic distributions for these
lda_bow_water = model[bow_water]
lda_bow_finance = model[bow_finance]
lda_bow_bank = model[bow_bank]
from gensim.matutils import kullback_leibler

print(kullback_leibler(lda_bow_water, lda_bow_bank))
print(kullback_leibler(lda_bow_finance, lda_bow_bank))
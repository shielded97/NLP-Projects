# Aidan Lambrecht
# COMP 7970
# Assignment 2
# second try since the first is wacky

from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_non_alphanum, strip_numeric, strip_punctuation, strip_short, strip_tags
import pandas as pd
import numpy as np
import gensim
import pickle
from scipy import spatial
import nltk
from nltk.tokenize import RegexpTokenizer

#------------------------------------------------------
# model
#------------------------------------------------------
class word2vec():
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 10
        self.window_size = 2
        self.embedding_size = 300
    
    def make_dictionary_data(self, reviews, save):
        word_index= dict()
        index_word = dict()
        word_counts = dict()
        corpus = []
        count = 0

        for review in reviews:
            for word in review:
                corpus.append(word)
                if word_index.get(word)==None:
                    word_index[word] = count
                    index_word[count] = word
                    count += 1
        self.vocab_size = len(word_index)

        if save == 1:
            #------------------------------------------------------
            # Save dictionary data
            #------------------------------------------------------
            word_index_file = open('word_to_index.txt', 'wb')
            pickle.dump(word_index, word_index_file)
            word_index_file.close()
            index_word_file = open('index_to_word.txt', 'wb')
            pickle.dump(index_word, index_word_file)
            index_word_file.close()
            corpus_file = open('corpus.txt', 'wb')
            pickle.dump(corpus, corpus_file)
            corpus_file.close()
            #------------------------------------------------------

        return word_index, index_word, corpus

    # apply part-of-speech tagger
    # parameters:
    #   num_of_reviews: number of examples
    #   reviews: corpus of documents
    def tagger(self, num_of_reviews, reviews):
        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(num_of_reviews):
            reviews[i] = str(reviews[i]).lower()  
            reviews[i] = tokenizer.tokenize(reviews[i])
            reviews[i] = nltk.pos_tag(reviews[i])
        return reviews

    # converts tags into usable tags for nltk lemmatization
    # input is one nltk regexpTokenizer part of speech tag
    def tag_converter(self, tag):
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

    def lemma(self, num_of_reviews, reviews):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for i in range(num_of_reviews):
            wordnet_tagged = map(lambda x: (x[0], self.tag_converter(x[1])), reviews[i])
            reviews[i] = []
            for word, tag in wordnet_tagged:
                if tag is None:
                    reviews[i].append(lemmatizer.lemmatize(word))
                else:
                    reviews[i].append(lemmatizer.lemmatize(word, tag))
        return reviews

    # loads model from saved weights
    # Parameters:
    #   infile_name: name of input hidden weights file (str)
    #   outfile_name: name of output hidden weights file (str)
    def restore_model(self, infile_name, outfile_name):
        infile = open(infile_name, 'rb')
        w_in = pickle.load(infile)
        infile.close()
        outfile = open(outfile_name, 'rb')
        w_out = pickle.load(outfile)
        outfile.close()
        self.hidden_input = w_in
        self.hidden_output = w_out
        print('Model restored successfully. . .\n')

    # this version makes one one-hot vector per word given
    def make_one_hots(self, target, word_index):
        vector = np.zeros(len(word_index))
        index = word_index[target]
        vector[index] = 1
        return vector

    def make_training_data(self, corpus, word_to_index, length_of_corpus, save):
        data = []
        for i in range(0, length_of_corpus - 1):
            context_words = []
            if i == 0: # first word
                for j in range(1, self.window_size + 1):
                    context_words.append(corpus[j])
            elif i == length_of_corpus - 2: # last word
                for j in range(length_of_corpus - 2 - self.window_size, length_of_corpus - 2):
                    context_words.append(corpus[j])
            else: # all other words
                for j in range(i - self.window_size, i): # context words before target
                    if j >= 0:
                        context_words.append(corpus[j])
                for j in range(i + 1, i + 1 + self.window_size): # context words after target
                    if j < length_of_corpus:
                        context_words.append(corpus[j])

            target_vector = self.make_one_hots(corpus[i], word_to_index)
            context_vector = []
            for word in context_words:
                context_vector.append(self.make_one_hots(word, word_to_index))
            data.append([target_vector, context_vector])
            if i % 1000 == 0:
                print('%d vectors made. . .'% i)

        if save == 1:
            #------------------------------------------------------
            # Save training data
            #------------------------------------------------------
            training_data_file = open('training_data.txt', 'wb')
            pickle.dump(np.array(data), training_data_file)
            training_data_file.close()
            print('Training data save: successful...')
            #------------------------------------------------------
        return np.array(data)#, dtype=object)

    # softmax function, defined in any number of websites, but in particular https://www.python-course.eu/softmax.php
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        s_max = exp_x / exp_x.sum(axis=0)    
        return s_max

    # forward propagation
    def forward(self, target_vector):
        # Calculate hidden layer  
        hidden_layer = np.dot(self.hidden_input.T, target_vector)
        # Calculate dense layer output
        u = np.dot(self.hidden_output.T, hidden_layer)
        # Run softmax
        predicted = self.softmax(u)

        return predicted, hidden_layer, u

    # backward propagation
    def backward(self, total_error, hidden_layer, target_vector):
        # Calculate hidden output and input
        hidden_input_dl = np.outer(target_vector, np.dot(self.hidden_output, total_error.T))
        hidden_output_dl = np.outer(hidden_layer, total_error)
        # Update weights
        self.hidden_input = self.hidden_input - (self.learning_rate * hidden_input_dl)
        self.hidden_output = self.hidden_output - (self.learning_rate * hidden_output_dl)

    # model training
    def train(self, data, save):
        self.vocab_size = len(word_index)
        self.hidden_input = np.random.uniform(-1, 1, (self.vocab_size, self.embedding_size))
        self.hidden_output = np.random.uniform(-1, 1, (self.embedding_size, self.vocab_size))

        for i in range(self.epochs):
            self.loss = 0
            for target_vector, context_vector in data:
                prediction, hidden, u = self.forward(target_vector)
                error = np.sum([np.subtract(prediction, word) for word in context_vector], axis=0)
                self.backward(error, hidden, target_vector)
                self.loss += -np.sum([u[np.ndarray.tolist(word).index(1)] for word in context_vector]) \
                    + len(context_vector) * np.log(np.sum(np.exp(u)))
            print('Epoch:', i, "Loss:", self.loss)
        if save == 1:
            #pickle the weights so we don't need to train multiple times
            weights_file = open('weights_inp.txt', 'wb')
            pickle.dump(self.hidden_input, weights_file)
            weights_file.close()
            weights_file = open('weights_out.txt', 'wb')
            pickle.dump(self.hidden_output, weights_file)
            weights_file.close()
        
        return self.hidden_input, self.hidden_output

	# get closest words via cosine similarity
    def get_similar_words(self, word, num_words, index_word, word_index):
        self.vocab_size = len(word_index)
        # get target vector
        index = word_index[word]
        vector1 = self.hidden_input[index]
        similarities = dict()
        # get rest of vectors
        for i in range(self.vocab_size):
            vector2 = self.hidden_input[i]
            result = 1 - spatial.distance.cosine(vector1, vector2)
            word = index_word[i]
            similarities[word] = result
        sorted_words = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
        print('---------------------')
        print('Target word: ', sorted_words[0][0])
        print('---------------------')
        for word, similarity in sorted_words[1:num_words + 1]:
            print(word, similarity)


#------------------------------------------------------
#------------------------------------------------------
# Scripted code/main
#------------------------------------------------------
#------------------------------------------------------
# Set train = 1 if you want to train a new model
#   train = 0 if you want to load one
# Set save = 1 if you want to save your new model
#   save = 0 if you don't want to overwrite old weights
#------------------------------------------------------
train = 0
save = 1

#------------------------------------------------------
# Instaniate w2v model
#------------------------------------------------------
model = word2vec()

#------------------------------------------------------
# Hyperparameters
#------------------------------------------------------
epochs = 10
embedding_words = 50
window_size = 2
learning_rate = 0.01
num_of_reviews = 500
word_index = dict()
index_word = dict()
corpus = []

model.epochs = epochs
model.embedding_size = embedding_words
model.window_size = window_size
model.learning_rate = learning_rate
print('Hyperparameters set. . .\n')

#------------------------------------------------------
# Preprocessing
#   remove stopwords, to lowercase, remove numbers, punctuation, etc.
#------------------------------------------------------
if train == 1:
    #------------------------------------------------------
    # Read in reviews
    #------------------------------------------------------
    reviews = pd.read_csv('./Review_word2vec.csv')#, index_col=0)
    print('Reading in reviews. . .\nDropping useless columns. . .')

    # drop unneeded columns
    reviews = reviews.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', \
        'HelpfulnessDenominator', 'Score', 'Time', 'Summary'], axis=1)
    reviews = reviews.loc[:, ~reviews.columns.str.contains('^Unnamed')]

    # convert to list
    reviews = reviews.values.tolist()

    print('Preprocessing reviews. . .\n----------------------------------')
    my_filter = [lambda x: x.lower(), strip_non_alphanum, strip_punctuation, strip_multiple_whitespaces, \
        remove_stopwords, strip_numeric, strip_short, strip_tags]

    for i, review in enumerate(reviews):
        if i % 50000 == 0:
            print(str(i) + ' reviews preprocessed. . .')
        if i > num_of_reviews:
            break
        reviews[i] = gensim.parsing.preprocessing.preprocess_string(str(review), filters=my_filter)

    # apply part of speech tagger and lemmatizer
    reviews = model.tagger(num_of_reviews, reviews)
    reviews = model.lemma(num_of_reviews, reviews)

    print('----------------------------------\nPreprocessing complete. . .\n')

    #------------------------------------------------------
    # Make dictionary data
    #------------------------------------------------------
    word_index, index_word, corpus = model.make_dictionary_data(reviews[:num_of_reviews], save)
    #------------------------------------------------------
    # Make training data
    #------------------------------------------------------
    training_data = model.make_training_data(corpus, word_index, len(corpus), save)
    print('Training data created. . .')
    #------------------------------------------------------
    # Train model
    #------------------------------------------------------
    print('Training has begun!\n')
    weights_inp, weights_out = model.train(training_data, save)
    print('Training complete!\n')

#------------------------------------------------------
# Load dictionary data
#------------------------------------------------------
elif train == 0:
    word_index_file = open("word_to_index.txt", 'rb')
    word_index = pickle.load(word_index_file)
    word_index_file.close()
    index_word_file = open("index_to_word.txt", 'rb')
    index_word = pickle.load(index_word_file)
    index_word_file.close()
    corpus_file = open("corpus.txt", 'rb')
    corpus = pickle.load(corpus_file)
    corpus_file.close()
    print('Dictionary information compiled. . .')
    #------------------------------------------------------
    #------------------------------------------------------
    # Load training data
    #------------------------------------------------------
    training_file = open("training_data.txt", 'rb')
    training_data = pickle.load(training_file)
    training_file.close()
    print('Training data loaded. . .')
    #------------------------------------------------------
    #------------------------------------------------------
    # Restore model
    #------------------------------------------------------
    model.restore_model('weights_inp.txt', 'weights_out.txt')

#------------------------------------------------------
# Print stats about our corpus
#------------------------------------------------------
# print('Epochs: %d\nEmbedding Size: %d\nWindow Size: %d\nLearning Rate: %.3f\nNumber of Examples (Reviews): %d\nCorpus Size: %d\nVocabulary Size: \n'% \
#     (epochs, embedding_words, window_size, learning_rate, num_of_reviews, len(corpus)))#, vocab_size))

#------------------------------------------------------
# Inference
#------------------------------------------------------
model.get_similar_words('coffee', 10, index_word, word_index)
model.get_similar_words('pasta', 10, index_word, word_index)
model.get_similar_words('tuna', 10, index_word, word_index)
model.get_similar_words('cookie', 10, index_word, word_index)

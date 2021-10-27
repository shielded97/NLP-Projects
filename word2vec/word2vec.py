# implement word2vec without libraries that train word2vec
from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_non_alphanum, strip_numeric, strip_punctuation, strip_short, strip_tags
import pandas as pd
import numpy as np
import gensim
import sklearn
import pickle

#------------------------------------------------------
# Function Definitions
#------------------------------------------------------

# make the dictionaries that we use
def make_dict_stuff(reviews):
    word_index= dict()
    index_word = dict()
    corpus = []
    count = 0

    for review in reviews:
        for word in review:
            corpus.append(word)
            if word_index.get(word)==None:
                word_index[word] = count
                index_word[count] = word
                count += 1
    vocab_size = len(word_index)

    return word_index, index_word, corpus, vocab_size

# make one hot lines based off the provided data
def make_one_hots(target, context, vocab_size, word_index):
    target_vector = np.zeros(vocab_size)
    context_vector = np.zeros(vocab_size)
    target_vector[word_index.get(target)] = 1
    for word in context:
        context_vector[word_index.get(word)] = 1
    return target_vector, context_vector

# make the data for training our model
def make_training_stuff(corpus, window_size, vocab_size, word_to_index, length_of_corpus):
    data = []
    for i in range(0, length_of_corpus - 1):
        context_words = []
        if i == 0: # first word
            for j in range(1, window_size + 1):
                context_words.append(corpus[j])
        elif i == length_of_corpus - 2: # last word
            for j in range(length_of_corpus - 2 - window_size, length_of_corpus - 2):
                context_words.append(corpus[j])
        else: # all other words
            for j in range(i - window_size, i): # context words before target
                if j >= 0:
                    context_words.append(corpus[j])
            for j in range(i + 1, i + 1 + window_size): # context words after target
                if j < length_of_corpus:
                    context_words.append(corpus[j])

        target_vector, context_vector = make_one_hots(corpus[i], context_words, vocab_size, word_to_index)
        data.append([target_vector, context_vector])   
    return data

# loss is defined according to http://www.claudiobellei.com/2018/01/06/backprop-word2vec/
def find_loss(u, context_words):
    sum = 0
    for j in np.where(context_words == 1)[0]:
        sum += u[j]
    sum = -sum
    num_context_words = len(context_words)
    loss = sum + (num_context_words * np.log(np.sum(np.exp(u))))
    # temp = -temp
    # temp1 = np.log(np.sum(np.exp(u)))
    # temp2 = len(np.where(context_words == 1)[0]) * temp1
    # sum = -np.sum([u[word.index(1)] for word in context_words])
    return loss

# softmax function, defined in any number of websites, but in particular https://www.python-course.eu/softmax.php
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    s_max = exp_x / exp_x.sum(axis=0)    
    return s_max

# forward propagation
def forward(hidden_input, hidden_output, target_vector):
    
    # Calculate hidden layer  
    hidden_layer = np.dot(hidden_input.T, target_vector)
    
    # Calculate dense layer output
    u = np.dot(hidden_output.T, hidden_layer)
    
    # Run softmax
    predicted = softmax(u)
    
    return predicted, hidden_layer, u

# backward propagation
def backward(hidden_input, hidden_output, total_error, hidden_layer, target_vector, learning_rate):
    
    # Calculate hidden output and input
    hidden_input_dl = np.outer(target_vector, np.dot(hidden_output, total_error.T))
    hidden_output_dl = np.outer(hidden_layer, total_error)
    
    # Update weights
    hidden_input = hidden_input - (learning_rate * hidden_input_dl)
    hidden_output = hidden_output - (learning_rate * hidden_output_dl)
    
    return hidden_input, hidden_output

# error calculations
# def error(prediction, context):
#      error = np.sum([np.subtract(prediction, word) for word in context], axis=0)
#      return error
def error(y_pred,context_words):
    
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}
    
    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update ( {index : 'yes'} )
        
    number_of_1_in_context_vector = len(index_of_1_in_context_words)
    
    for i,value in enumerate(y_pred):
        
        if index_of_1_in_context_words.get(i) != None:
            total_error[i]= (value-1) + ( (number_of_1_in_context_vector -1) * value)
        else:
            total_error[i]= (number_of_1_in_context_vector * value)
            
    return  np.array(total_error)

# training function
def train(training_data, embedding_words, learning_rate, vocab_size):
    
    # prepare variables to save weights
    weights_inp = []
    weights_out = []

    # randomize initial weights to learn from 
    hidden_input = np.random.uniform(-1, 1, (vocab_size, embedding_words))
    hidden_output = np.random.uniform(-1, 1, (embedding_words, vocab_size))

    for i in range(epochs):
        print('Epoch %d has begun'% (i + 1))
        loss = 0
        for targ, ctx in training_data:
            # need to forward propagate
            predicted, hidden_layer, u = forward(hidden_input, hidden_output, targ)
            # calculate error
            err = error(predicted, ctx)
            # need to back propagate to update weights
            hidden_input, hidden_output = backward(hidden_input, hidden_output, err, hidden_layer, targ, learning_rate)
            # get loss
            temp = find_loss(u, ctx)
            loss = loss + temp
        # save weights for further use
        weights_inp.append(hidden_input)
        weights_out.append(hidden_output)
        w_arr_inp = np.array(weights_inp)
        w_arr_out = np.array(weights_out)

        # show some updates
        print('Epoch: %d\n\t Loss: %f\n' %(i + 1, loss))

    return w_arr_inp, w_arr_out

#inference works via cosine similarity
def cosine_similarity(word_index, index_word, target, vocab_size, weights):
    
    similarities = dict()
    # calculate for each word in relation to this the target
    for i in range(vocab_size):
        cos_sim = sklearn.metrics.pairwise.cosine_similarity(weights[word_index[target]], weights[i])
        word = index_word[i]
        similarities[word] = cos_sim

    return similarities

#------------------------------------------------------
# Scripted code/main
#------------------------------------------------------
reviews = pd.read_csv('./Review_word2vec.csv')#, index_col=0)
print('Reading in reviews. . .\nDropping useless columns. . .')

# drop unneeded columns
reviews = reviews.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', \
    'HelpfulnessDenominator', 'Score', 'Time', 'Summary'], axis=1)
reviews = reviews.loc[:, ~reviews.columns.str.contains('^Unnamed')]

# convert to list
reviews = reviews.values.tolist()

#------------------------------------------------------
# Hyperparameters
#------------------------------------------------------
epochs = 10
embedding_words = 300
window_size = 2
learning_rate = 0.01
num_of_reviews = 1 # len(reviews) - 1
word_index = dict()
index_word = dict()
corpus = []
vocab_size = 0
print('Hyperparameters set. . .\n')

#------------------------------------------------------
# preprocessing
# remove stopwords, to lowercase, remove numbers, punctuation, etc.
#------------------------------------------------------
print('Preprocessing reviews. . .\n----------------------------------')
my_filter = [lambda x: x.lower(), strip_non_alphanum, strip_punctuation, strip_multiple_whitespaces, \
    remove_stopwords, strip_numeric, strip_short, strip_tags]
for i, review in enumerate(reviews):
    if i % 50000 == 0:
        print(str(i) + ' reviews preprocessed. . .')
    if i > num_of_reviews:
        break
    reviews[i] = gensim.parsing.preprocessing.preprocess_string(str(review), filters=my_filter)

print('----------------------------------\nPreprocessing complete. . .\n')

word_index, index_word, corpus, vocab_size = make_dict_stuff(reviews[:num_of_reviews])
print('Dictionary information compiled. . .')
training_data = make_training_stuff(corpus, window_size, vocab_size, word_index, len(corpus))
print('Training data created. . .')

for i in range(len(training_data)):
    if (i == 0) or (i == (len(training_data) - 1)) or (i == round((len(training_data)/2))):
        print('Target vector:   %s ' %(training_data[i][0]))
        print('Context  vector: %s \n' %(training_data[i][1]))

#------------------------------------------------------
# Print stats about our corpus
#------------------------------------------------------
print('Epochs: %d\nEmbedding Size: %d\nWindow Size: %d\nLearning Rate: %.3f\nNumber of Examples (Reviews): %d\nCorpus Size: %d\nVocabulary Size: %d\n'% \
    (epochs, embedding_words, window_size, learning_rate, num_of_reviews, len(corpus), vocab_size))
print('Training has begun!\n')
weights_inp, weights_out = train(training_data, embedding_words, learning_rate, vocab_size)

# pickle the weights so we don't need to train multiple times
weights_file = open('weights_inp.txt', 'wb')
pickle.dump(weights_inp, weights_file)
weights_file.close()
weights_file = open('weights_out.txt', 'wb')
pickle.dump(weights_out, weights_file)
weights_file.close()

print('done')
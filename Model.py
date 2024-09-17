import random
import json
import pickle #ibrary in Python is used for serializing and deserializing Python objects.
# Using pickle to save the processed data allows for easy retrieval and reuse without the need to preprocess the data again.
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # for neural network model
import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer() #initilizing the Wordnet lemmatizer

intents = json.loads(open('"*********Enter the Intents.json File path ****************" ').read()) # loading form intents file

words = [] # unique words in the pattern, used as features in the training model
classes = []  # unique tags, used as label when we tarin the model
documents = [] # A list of tuples, where each tuple contains a tokenized pattern and its associated tag.
ignoreLetters = ['?', '!', '.', ',']
# iterate through each intents
for intent in intents['intents']:
    # iterate through each pattern
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern) # tokenize each pattern into a list
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Lemmatization --> lemma, and ignore the punc
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words)) # sort, remove duplicates
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = [] # initializes an empty list to store the training data and each element is a training example.
outputlabel = [0] * len(classes)
#  Creates a list outputlabel of zeros with a length of the classes to represent the output label for each training instance.

for document in documents:
    bag = [] # bag represent the words in the document this representation will be used as input features for training the chatbot model
    WEx = document[0] # Extracts the tokenized words from the current document.
    # accesses the first element of the tuple, which contains the tokenized words.


    WEx = [lemmatizer.lemmatize(word.lower()) for word in WEx]
    #Lemmatizes and converts each word to lowercase. This ensures consistency and reduces the size of the vocabulary

    for word in words:
        bag.append(1) if word in WEx else bag.append(0)
# This creates a binary representation of the presence or absence of each word according to words and word pattern
    # in the vocabulary for the current document.

    outputRow = list(outputlabel)
    # Creates a copy of the outputlabel list to represent the output vector for the current document
    outputRow[classes.index(document[1])] =1 # document[1] represents the intent tag associated with the current document
    training.append(bag + outputRow)
    #This forms the complete training example. contains both the input features (bag of words representation) and
    # the output labels (one-hot encoded intent tags) for each training example.

random.shuffle(training)
# Shuffling ensures that the model sees a mixture of examples from all classes during each epoch
training = np.array(training) # make it as numpy array.

# splits the NumPy array into input (trainX) and output (trainY) arrays.
trainX = training[:, :len(words)] # extract the input features
trainY = training[:, len(words):]# extract the output label

# so here we are building the model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(len(trainX[0]),))) #the input layer expects vectors of length equal to the number of features
model.add(tf.keras.layers.Dense(128, activation='relu')) # Rectified Linear Unit  helps the network learn complex patterns by introducing non-linearity
#Neurons help in extracting and representing features from the input data.
model.add(tf.keras.layers.Dropout(0.5)) # Dropout helps prevent overfitting
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5)) # Further helps to prevent overfitting
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))
# softmax Turns  a vector z = [z1,z2,...,zk]of k arbitrary values into probabilities

# Configuration the model for training
# Stochastic Gradient Descent
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# used for multi-class classification problems
#The optimizer used to minimize the loss function during training.

# training the Model
Tmod = model.fit(np.array(trainX), np.array(trainY), epochs=100, batch_size=4, verbose=1)
model.save('my_model.keras',Tmod)

# recorded during the training process.

print('**********************')
print('The Model has Trained')




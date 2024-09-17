import random
import json
import pickle
import numpy as np
import nltk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from nltk.stem import WordNetLemmatizer
from keras.models import load_model




lemmatizer = WordNetLemmatizer()




intents = json.loads(open('*******************"Enter the Intents.json file Path here"**************').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('my_model.keras')




# This function tokenizes and lemmatizes the input sentence.
# 'How', 'are'...
# Tokenize and lemmatize.
def To_Le(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def collection(sentence):
    sentence_words = To_Le(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def Pre(sentence):
    bow = collection(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # the results are sorted in descending order based on the probability scores.
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    # assumes that the list is sorted in descending order of probability, so the first item usually represents the most likely intent
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("GO! Bot is running!")


while True:
    message = input("\nMe: ")
    if message.lower() in ['see you', 'bye', 'have a nice day']:
        print("Bot: Goodbye!")
        break

    ints = Pre(message)
    res = response(ints, intents)
    print("Bot:", res)

import re
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

ignore_words = ['?', '!']


class ChatBot:

    def __init__(self):
        self.model = None
        self.words = []
        self.classes = []
        self.documents = []
        self.lemmatizer = WordNetLemmatizer()
        data_file = open("intents.json").read()
        self.intents = json.loads(data_file)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def data_preprocessing(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:

                # tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)

                # add documents in the corpus
                self.documents.append((w, intent['tag']))

                # add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # lemmatize, lower each word and remove duplicates
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_words]
        self.words = sorted(list(set(self.words)))

        # sort classes
        self.classes = sorted(list(set(self.classes)))

    def bow(self, sentence):
        bag = [0] * len(self.words)
        for s in self.clean_up_sentence(sentence):
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence, model):
        p = self.bow(sentence)
        res = model.predict(np.array([p]))[0]
        error = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, text):
        tag = self.predict_class(text, self.model)[0]['intent']
        result = None
        for i in self.intents['intents']:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        if result is not None:
            return result
        else:
            return "Didn't get your question"

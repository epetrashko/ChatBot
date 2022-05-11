import re
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer

ignore_words = ['?', '!']


class ChatBot:

    def __init__(self):
        self.model = None
        self.intents_json = None

        self.lemmatizer = WordNetLemmatizer()

        self.words = []
        self.classes = []
        self.documents = []
        data_file = open("intents.json").read()
        self.intents = json.loads(data_file)

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

    def predict_class(self, text, model):
        pass

    def get_response(self, text):
        ints = self.predict_class(text, self.model)
        tag = ints[0]['intent']
        list_of_intents = self.intents_json['intents']
        result = None
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        if result is not None:
            return result
        else:
            return "Didn't get your question"

import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


class ChatBot:
    def __init__(self):
        self.model = None
        self.intents_json = None
        self.words = None
        self.classes = None
        self.list_of_intents = None

    def bow(self, sentence):
        bag = [0] * len(self.words)
        for s in clean_up_sentence(sentence):
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
        # list_of_intents = self.intents_json['intents']
        tag = self.predict_class(text, self.model)[0]['intent']
        result = None
        for i in self.list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        if result is not None:
            return result
        else:
            return "Didn't get your question"

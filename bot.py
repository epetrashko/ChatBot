import re
import random


class ChatBot:
    def __init__(self):
        self.model = None
        self.intents_json = None


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

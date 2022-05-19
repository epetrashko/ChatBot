import json
import random

from bot import ChatBot


class ModelTest:
    def __init__(self, rebuild):
        self.intents = json.loads(open("intents.json").read())
        if rebuild:
            self.chat_bot = ChatBot(intents=self.intents, rewrite=True, error_threshold=0.75)
        else:
            self.chat_bot = ChatBot(intents=self.intents)

    def check_respond(self):
        intents_passed = 0
        for intent in self.intents["intents"]:
            for q in intent["patterns"]:
                a = self.chat_bot.get_response(q)
                print(f"Intent: {q},\nQuestion: {q},\nAnswer: {a}\n\n")
                assert a in intent["responses"], f"Intent: {q},\nQuestion: {q},\nAnswer: {a}\n"
            intents_passed += 1
            print(f"{intents_passed} passed")

    def check_ignore_words_independence(self):
        for intents_passed in range(10):
            intent = random.choice(self.intents["intents"])
            q = random.choice(intent["patterns"])
            q += "?!"
            a = self.chat_bot.get_response(q)
            assert a in intent["responses"], f"Intent: {q},\nQuestion: {q},\nAnswer: {a}\n"
            print(f"{intents_passed + 1} passed")


tests = ModelTest(True)
tests.check_respond()
tests.check_ignore_words_independence()

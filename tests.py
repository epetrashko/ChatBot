import json
import random

from bot import ChatBot


class ModelTest:
    def __init__(self, rebuild):
        self.intents = json.loads(open("intents.json").read())
        self.chat_bot = ChatBot(intents=self.intents, rewrite=rebuild, error_threshold=0.75)

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

    def manual_testing(self):
        while True:
            user_input = input("User(quit for terminating): ")
            if not user_input.lower() == "quit":
                a = self.chat_bot.get_response(user_input)
                print(f"Bot:{a}\n")
            else:
                break


tests = ModelTest(rebuild=True)
tests.check_respond()
tests.check_ignore_words_independence()
tests.manual_testing()

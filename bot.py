import os
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import gradient_descent_v2
from tensorflow.python.keras.models import load_model

ignore_words = ['?', '!']


class ChatBot:

    def __init__(self, intents: dict, error_threshold=0.75, rewrite=False):
        self.model = None
        self.__words = []
        self.__labels = []
        self.__pattern_to_tag = []
        self.__error_threshold = error_threshold
        self.__lemmatizer = WordNetLemmatizer()
        self.__intents = intents
        self.__data_preprocessing()
        self.__load_model(rewrite=rewrite)

    def __load_model(self, rewrite):
        if rewrite or not os.path.exists("model.h5"):
            self.__create_training_data()
        else:
            self.model = load_model('model.h5')
            print("\nLoading model")

    def __data_preprocessing(self):
        for intent in self.__intents['intents']:
            for pattern in intent['patterns']:

                # tokenize each word
                words = nltk.word_tokenize(pattern)
                self.__words.extend(words)

                # add documents in the corpus
                self.__pattern_to_tag.append((words, intent['tag']))

                # add to our classes list
                if intent['tag'] not in self.__labels:
                    self.__labels.append(intent['tag'])

        # lemmatize, lower each word and remove duplicates
        self.__words = [self.__lemmatizer.lemmatize(word.lower()) for word in self.__words if word not in ignore_words]
        self.__words = sorted(list(set(self.__words)))

        # sort labels
        self.__labels = sorted(list(set(self.__labels)))

    def __create_training_data(self):
        # create our training data
        training = []

        # training set, bag of words for each sentence
        for pattern, tag in self.__pattern_to_tag:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = pattern

            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self.__lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # create our bag of words array with 1, if word match found in current pattern
            for word in self.__words:
                bag.append(1) if word in pattern_words else bag.append(0)

            output = [0] * len(self.__labels)
            output[self.__labels.index(tag)] = 1
            training.append([bag, output])

        # shuffle features and converting it into numpy arrays
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        x_train = list(training[:, 0])
        y_train = list(training[:, 1])

        print("Training data created")
        self.__create_model(x_train, y_train)

    def __create_model(self, x_train, y_train):
        # Create NN model to predict the responses
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(y_train[0]), activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this
        # model
        sgd = gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # fitting and saving the model
        hist = self.model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
        self.model.save('model.h5', hist)
        print("\nModel Created Successfully!")

    def __clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.__lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def __bow(self, sentence):
        bag = [0] * len(self.__words)
        for s in self.__clean_up_sentence(sentence):
            for i, w in enumerate(self.__words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def __predict_tag(self, sentence):
        p = self.__bow(sentence)
        res = self.model.predict(np.array([p]))[0]
        results = [[i, r] for i, r in enumerate(res) if r > self.__error_threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.__labels[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, text):
        tags = self.__predict_tag(text)
        if len(tags) == 0:
            return None
        tag = tags[0]['intent']
        for i in self.__intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])

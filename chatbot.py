import numpy
import tflearn
import tensorflow
import json
import random
import pickle
import nltk
import tensorflow as tf
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("botData.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docsX = []
    docsY = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docsX.append(wrds)
            docsY.append(intent["tag"])
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []
    outEmpty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docsX):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        outputRow = outEmpty[:]
        outputRow[labels.index(docsY[x])] = 1
        training.append(bag)
        output.append(outputRow)

    training = numpy.array(training)
    output = numpy.array(output)
    with open("botData.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load("chatbotModel.tflearn")

except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("chatbotModel.tflearn")

def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]
    sWords = nltk.word_tokenize(s)
    sWords = [stemmer.stem(word.lower()) for word in sWords]
    for se in sWords:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You:")
        if inp.lower() == "quit":
            break

        results = model.predict([bagOfWords(inp, words)])[0]
        resultsIndex = numpy.argmax(results)
        tag = labels[resultsIndex]
        if results[resultsIndex] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't get that. Try again!")
chat()


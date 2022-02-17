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


with open("intents.json") as file:   # loads in the json file with the responses
    data = json.load(file)

#print(data["intents"])     # prints out the json file        print(data["intents"]) prints out everything in the intents list (dictionares)

try:
    with open("botData.pickle", "rb") as f:      # rb stands for reads bytes because we are going to save the data as bytes
        words, labels, training, output = pickle.load(f)        # saves the words, labels, training, output variables into the pickle file
except:
    # loop through the patterns and responses and see what group/tag they are in
    words = []
    labels = []
    docsX = []
    docsY = []
    # made 2 lists for docs (docsX and docsY) because for each pattern we also want to put another element in docsY that stands for what intent its apart of / what tag it is apart of

    for intent in data["intents"]:      # loops through all the dictionaries
        for pattern in intent["patterns"]:      # accesses the patterns             # stemming is going to take each word that in our pattern and bring it down to the root word. for example: "Is anyone there?". If we are looking at the word "there?". The roots word of "there?" is just "there"       by stemming we are making the bot be more accurate
            wrds = nltk.word_tokenize(pattern)   # is going to outputs a list with all teh different words in it                  To get all our different words in our pattern and to stem them we need to do something called tokenize. Tokenize is to get all the words in our pattern
            words.extend(wrds)    # going to put all the tokenize words into the words list (line 17). The reason we do this si because we want to know all the different words that are in our intents.json file
            docsX.append(wrds)    # we are going to add to the docs list (line 19) our pattern of words
            docsY.append(intent["tag"])     # each entry in docsX corresponds an entry in docsY     This is important for identification and training the model

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # we are going to stem all of the words in the words list and remove any duplicate elements so we can figure out what our vocabulary size of the model is       how many words it has seen already
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]           # w.lower() converts all letters to lowercase
    words = sorted(list(set(words)))    # makes our words list into a set that removes all duplicates       set() : takes in all the words and makes sure there is no duplicates
    labels = sorted(labels)

    # our training and testing output
    # all our data is in strings which isn't going to work for our neural network because neural networks only understands numbers. So we are going to create what is known as a "bag of words" that represents all of the words in any given pattern and we are going to use that to train our model.
    # a "bag of words" is whats known as "one hot encoded" which mean that essentially we are going to have a list [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1] like this that is going to be the length of the amount of words that we have. For example if we had 100 words then each encoding is going to have 100 entries that is going to be zeros and ones
    # [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1] each position in this list will represent either a words exists or doesn't exist, also the numbers in the list could be 2,5,6, etc. It just tells you how many times that each word occurs.
    # When we encode this we are going to say the first word in the list is "a" the second word is "bite" and the third words is "goodbye". So we are going to look at a sentence and we are going to encode it in the form of a list ( [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1] ). E.g if there is 2 "a" then we say [2,0,0,0,0,0,0,0,1,1,1,1,1,1,1]. If the words "bite" exists in our sentence we put in how many times it is in the sentence, if it doesn't exist then we put in a zero

    # Our input is going to be a big list with how many word entries we have in it and it is just going to say whether a words exists or it doesn't
    training = []
    output = []

    outEmpty = [0 for _ in range(len(labels))]         # Our output also has to be in a different form as well         the output is all of our tags(greeting, age, name, etc). We need to turn these into one hot encoded as well which means we are going to have a list that has a zero in it for all of the different classes for how many classes that we have and if that class/if that tag exists or is the one that we want we will put a 1 there
                                                    # [0,0,0,1]     1st tag: "hi"   2nd tag: "buy"  3rd tag: "sell"   4th tag: "help"       this means that the tag help is associated with the input that we have here because it has the 1 beside it

    for x, doc in enumerate(docsX):
        bag = []        # going to be our "bag of words" / one hot encoded

        wrds = [stemmer.stem(w) for w in doc]        # going to stem all of our words that are in the patterns beacuse when we stem them we only stem each word in our words list, we didn't stem them when we added them into docsX so we are going to stem them here

        for w in words:    # we are going to loop through all the different words that are in our wrds list that are stemmed (line54) and we are going to put either a 1 or 0 into our "bag of words" depending on if its in the main words list (line 35) or not
            if w in wrds:       # if the word exists in the current wrds pattern we are looping through (line54)
                bag.append(1)       # which means the word exists here ( 1 representing that this word exists)

            else:
                bag.append(0)       # which means the word doesn't exists here ( 0 representing that this doesn't word exists)

        # Now we are going to generate the output and append these into training and output (lines 45 and 46)
        # the output has to be generated like this [0 for _ in range(len(classes))] where we have a bunch of 0's and 1's representing the tag that
        outputRow = outEmpty[:]
        outputRow[labels.index(docsY[x])] = 1
        # we are going to look through the labels list (line18), we are going to see where the tag is in that list and then we are going to set that value to 1 in our output row

        training.append(bag)
        output.append(outputRow)
        # we are going to have 2 lists: a training list that is going to have a bunch of "bags of words" which are like a list of 0's and 1's. A list of outputs that are again a list of 0'a and 1's. Both lists are one hot encoded

    # going to turn the training and output lists into np.arrays because we have to work with numpy arrays for tflearn
    training = numpy.array(training)
    output = numpy.array(output)

    with open("botData.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# creating the model using tflearn
tf.compat.v1.reset_default_graph()

# neural network layers
net = tflearn.input_data(shape=[None, len(training[0])])        # this is going to define the input shape we are expecting for our model
net = tflearn.fully_connected(net, 8)
# going to add this fully connected layer to our neural network which starts at our input data (line79) and we are going to have 8 neurons for that hidden layer
net = tflearn.fully_connected(net, 8)           # another hidden layer that has 8 neurons as well
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")        # our output layer      softmax allows us to get probabilities for each output
net = tflearn.regression(net)

# training the model
model = tflearn.DNN(net)    # DNN is just a type of neural network

#line 80 to line 87 is our completed model
# Started with an input data (line80) which is the length of the training data Two hidden layers with 8 neurons in each layer fully connected. That is also connected to a output layer that has neurons representing each of our classes

try:
    model.load("chatbotModel.tflearn")

except:
    # fit the model
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)     # start passing in our training data      n_epoch is the number of times that the model is going to see the same data     show_metric=True so we get a nice output when we are fitting the model
    model.save("chatbotModel.tflearn")      #save the model

# making predictions. turning a sentence inputted by the user into a "bag of words"
def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]        # creates a blanket list of words and then we will change the elements in the list if the word is in the list or not

    sWords = nltk.word_tokenize(s)      # creates a list of tokenized words
    sWords = [stemmer.stem(word.lower()) for word in sWords]        # stems all of the words

    for se in sWords:
        for i, w in enumerate(words):
            if w == se:     # the current word we are looking at in the words list is equal to the word in the se(sentence)
                bag[i] = 1

    return numpy.array(bag)

# writing the code that will ask the user for an input/sentence
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You:")     # inp stands for input      you: means that you type to the bot
        if inp.lower() == "quit":
            break
        # if they didn't type quit then we are going to turn the inp(input) into a "bag of words", feed it to the model and get what the models response should be
        results = model.predict([bagOfWords(inp, words)])[0]
        resultsIndex = numpy.argmax(results)        # will give us the index of the greatest value in the list
        tag = labels[resultsIndex]      # gives us the label that the model thinks the message is

        if results[resultsIndex] > 0.7:     # if the bot has at least 70% confidence

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))

        else:
            print("I didn't get that. Try again!")
chat()

























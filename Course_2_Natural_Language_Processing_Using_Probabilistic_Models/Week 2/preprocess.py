# Importing Libraries
import nltk
from nltk.corpus import treebank
import re
import pprint
import numpy as np
import pandas as pd
import time
import random
import sklearn
from sklearn.model_selection import train_test_split

class PreprocessData(object):
    def __init__(self):
        self.tagged_sentences = list()
        self.train_data = None
        self.test_data = None

    # function to fetch tagged sentences...
    def getTaggedSentences(self):
        print("Fetching Tagged Sentences")
        tagged_sentences = list(treebank.tagged_sents())
        self.tagged_sentences = tagged_sentences
        print("Tagged Sentences are fetched.")
        print()


    # function to split the data into train and test dataset....
    def splitData(self):
        print("Splitting the data into train and test....")
        random.seed(1234)
        # Split the data into train and test....
        train_data, test_data = train_test_split(self.tagged_sentences, test_size = 0.3)
        self.train_data = train_data
        self.test_data = test_data
        print("Data is split into train and test....")
        print()
        print("Length of Corpus : ", len(self.tagged_sentences))
        print("Length of Train Data : ", len(self.train_data))
        print("Length of Test Data : ", len(self.test_data))


    # func to get list of tagged words and their respective tags...
    def getTaggedTokens(self):
        # function to get tagged sentences...
        self.getTaggedSentences()
        # function to split the data into train and test...
        self.splitData()

        train_tagged_words = []
        tokens = []

        # get the word-tag pair....
        train_tagged_words = [tup for sentence in self.train_data for tup in sentence]
        print("Length of Train-Tagged Words : ", len(train_tagged_words))
        # get the tokens...
        tokens = [pair[0] for pair in train_tagged_words]
        print("Length of the tokens are : ", len(tokens))

        # get the vocabulary...
        vocab = set(tokens)
        print("Length of the Vocab is : ", len(vocab))

        # get the tags...
        tags = set([pair[1] for pair in train_tagged_words])
        print("Length of the tags are : ", len(tags))

        return train_tagged_words, vocab, list(tags)

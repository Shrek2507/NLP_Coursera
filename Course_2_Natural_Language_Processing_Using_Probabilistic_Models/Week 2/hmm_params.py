import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import re

class CreateHMMParams(object):
    def __init__(self):
        self.text_data = []
        self.word_tags_list = list()
        self.punct = '.,;:'
        self.tag_count = dict()
        

    # Read the Text file....
    def read_file(self, file_path):
        # Open the file...
        print("Reading the contents of the file...")
        with open(file_path) as fp:
            text_data = fp.readlines()
        
        print("File Contents are read....")

        # clean the text data...
        for text in text_data:
            # Remove all special characters....
            clean_text = re.sub('\n','', text).strip()

            self.text_data.append(clean_text)

        self.text_data = ' '.join(self.text_data)


    # function to create Word Tag pair....
    def createWordTagPair(self):
        tokenized_words = word_tokenize(self.text_data)  # tokenize the sentence into words...
        print("Tokenized Words are : ")
        print(tokenized_words)

        # Loop to create word-tag pairs...
        for token in tokenized_words:
            word_tag = tuple()
            # check if the token is a punctuation....
            if token in self.punct:
                pass
            else:
                word, tag = token.split('/')
                word_tag = (str(word), str(tag))
                self.word_tags_list.append(word_tag)
        
        print("The Word Tag Pairs are : ")
        print(self.word_tags_list)

    # func to get the count of tags....
    def getTagCounts(self):
        tag_counts = {}
        for pair in self.word_tags_list:
            tag = pair[1]

            if tag in tag_counts.keys():
                tag_counts[tag]+=1
            else:
                tag_counts[tag] = 1
                
        self.tag_count = tag_counts
    


file_path = 'Corpus_Text.txt'
obj_hmm = CreateHMMParams()

# Read the file...
obj_hmm.read_file(file_path)

# Create Word-Tag Pairs....
obj_hmm.createWordTagPair()

# get tag counts...
obj_hmm.getTagCounts()


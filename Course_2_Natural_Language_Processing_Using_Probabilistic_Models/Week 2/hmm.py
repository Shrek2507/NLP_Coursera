import numpy as np
import pandas as pd
from preprocess import PreprocessData

class ViterbiException(Exception):
    pass

class Viterbi(object):
    def __init__(self):
        obj_preprocess = PreprocessData()
        self.train_tagged_words, self.vocab, self.tags = obj_preprocess.getTaggedTokens()
        self.test_tagged_words = obj_preprocess.test_data
        self.word_given_tag = np.zeros((len(self.tags), len(self.vocab)))
        self.tags_matrix = np.zeros((len(self.tags), len(self.tags)), dtype='float32')

    # function to get word given tag: Emission Probability
    def getEmissionProbability(self, word, tag):
        emission_prob = 0.0
        train_bag = self.train_tagged_words
        # collect list of all word-tag list which belongs to input tag...
        tag_list = [pair for pair in train_bag if pair[1] == tag]
        count_tag = len(tag_list)
        # collect list of all word-tag list which belongs to input word...
        word_given_tag_list = [pair for pair in train_bag if pair[0] == word]
        count_word_given_tag_list = len(word_given_tag_list)

        emission_prob = round(float(count_word_given_tag_list)/float(count_tag), 4)

        return emission_prob

    # function to implement transition probability transition from tag1 ---> tag2
    def getTransitionProbability(self, tag2, tag1):
        count_tag1=0
        count_tag1_tag2 = 0
        transition_prob = 0.0
        train_bag = self.train_tagged_words
        tags = [pair[1] for pair in train_bag]
        
        # fetch a list of tags == tag1
        list_tag1 = [t for t in tags if t==tag1]
        count_tag1 = len(list_tag1)

        # loop to get a count of tags making a transition from t1-->t2
        for i in range(0, len(tags)-1):
            if tags[i] == tag1 and tags[i+1] == tag2:
                count_tag1_tag2+=1

        transition_prob = round(float(count_tag1_tag2)/float(count_tag1),4)

        return transition_prob


    # function to implement HMM algorithm....
    def implementViterbi(self):
        try:
            state = []
            words = self.test_tagged_words
            # Create a t x t Transition probability matrix....
            print("Creating transition probability matrix : ")
            for i, t1 in enumerate(list(self.tags)):
                for j, t2 in enumerate(list(self.tags)):
                    # get the transition probability and store it in the matrix...
                    self.tags_matrix[i,j] = self.getTransitionProbability(t2, t1)
            
            print("Transition probability matrix is created.")
            # create a dataframe of tags....
            print("Creating a Dataframe of Tags...")
            tags_df = pd.DataFrame(self.tags_matrix, columns = list(self.tags), index = list(self.tags))
            print("Dataframe of Tags is created...")
            print()
            print("Iterating through the Test set...")
            for key, word in enumerate(words):
                # initialize the probability column....
                prob = []
                for tag in self.tags:
                    if key == 0:
                        transition_prob = tags_df.loc['.', tag]
                    else:
                        transition_prob = tags_df.loc[state[-1], tag]
                
                    # compute emission and state probabilities....
                    emission_prob = self.getEmissionProbability(words[key], tag)
                    state_probability = emission_prob * transition_prob
                    prob.append(state_probability)
                
                pmax = max(prob)
                # getting the state for which prob is max...
                state_max = self.tags[prob.index(pmax)]
                state.append(state_max)

            print("Iteration is completed in test set.")

        except:
            raise ViterbiException()
        

obj_viterbi = Viterbi()
obj_viterbi.implementViterbi()


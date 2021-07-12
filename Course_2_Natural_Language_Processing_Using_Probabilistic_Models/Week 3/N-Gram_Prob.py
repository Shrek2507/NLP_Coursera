import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pprint


# Clean the text...
def cleanText(corpus):
    clean_corpus = corpus.replace("\n", "")

    # remove punctuation...
    clean_corpus = re.sub(r"\W+", " ", corpus)

    return clean_corpus.lower()


# func to get the word sequence...
def getWordSequence(tokens, word_1, word_2):
    countDict = {}
    countDict = dict(Counter(tokens))
    lengthOfCorpus = len(tokens)
    totalNGrams = 0
    probScore = []

    print("Tokens are : ")
    print(tokens)
    print()
    print('Length Of Tokens are : ', lengthOfCorpus)

    print("Count Dict is : ")
    pprint.pprint(countDict, indent=2)

    totalNGrams = len(word_tokenize(word_1.lower()+' '+word_2.lower()))

    # check for Unigram probability...
    if totalNGrams == 1:
        # calculate the probability score...
        probScore.append(countDict[word_1.lower()])
        probScore.append(lengthOfCorpus)

        return "/".join(list(map(str, probScore)))

    # check for Bigram probability...
    elif totalNGrams == 2:
        # get the count of Unigrams...
        countWord1 = countDict[word_1.lower()]

        # get bigrams count...
        countBigrams = dict(
            Counter(ngrams(tokens, len(word_tokenize(word_1.lower()+" "+word_2.lower())))))

        probScore.append(countBigrams.get(tuple(word_tokenize(word_1.lower(
        )+" "+word_2.lower()))))

        probScore.append(countWord1)

        return "/".join(list(map(str, probScore)))

    # Check for N-Gram probability...
    elif totalNGrams > 2:
        # get the count of N-Grams...
        # get N-Gram count...
        countNGrams = dict(
            Counter(ngrams(tokens, len(word_tokenize(word_1.lower()+" "+word_2.lower())))))

        # get the count N-1 Gram
        countWord1 = dict(Counter(
            ngrams(tokens, len(word_1.lower().split()))))

        probScore.append(countNGrams.get(tuple(word_tokenize(word_1.lower(
        )+" "+word_2.lower()))))

        probScore.append(countWord1.get(tuple(word_1.lower().split())))

        return "/".join(list(map(str, probScore)))


# function to calculate N-Gram Probability...
def getNGramProb(corpus, word_1, word_2):
    try:
        probScore = ""
        # clean the text....
        clean_corpus = cleanText(corpus)

        # tokenize the text...
        tokenizedCorpus = list(word_tokenize(clean_corpus))

        # get word sequences
        probScore = getWordSequence(tokenizedCorpus, word_1, word_2)

        print("The Probability Score is : ", probScore)

    except Exception as e:
        import traceback
        print(traceback.format_exc())


corpus = """In every place of great resort the monster was the fashion.
They sang of it in the cafes, ridiculed it in the papers, and represented it on the stage."""

word_1 = "it in the"
word_2 = "papers"
print("The Corpus is : ")
print(corpus)

print()
getNGramProb(corpus, word_1, word_2)

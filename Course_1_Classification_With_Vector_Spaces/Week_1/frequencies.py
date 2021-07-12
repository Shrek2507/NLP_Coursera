import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter

class Preprocess_Data(object):
	def __init__(self):
		self.corpus = None

	# function to extract the corpus....
	def extract_corpus(self, file_name, sheet_name):
		print("Extracting the corpus....")
		small_corpus = pd.read_excel(file_name, sheet_name)

		print("Corpus is extracted...")

		self.corpus = small_corpus

		positive_sentiments = self.corpus[self.corpus['Sentiment'] == 1]
		negative_sentiments = self.corpus[self.corpus['Sentiment'] == 0]

		positive_sentiments_sentences = list(positive_sentiments['Tweet'])
		positive_sentiments_values = list(positive_sentiments['Sentiment'])

		negative_sentiments_sentences = list(negative_sentiments['Tweet'])
		negative_sentiments_values = list(negative_sentiments['Sentiment'])

		return positive_sentiments_sentences, positive_sentiments_values, negative_sentiments_sentences, negative_sentiments_values

	# function to generate the vocab frequency....
	def generate_unique_vocabulary(self, positive_sentiments_sentences, negative_sentiments_sentences):
		# create vocab frequency list....
		vocab_frequency_positive = {}
		vocab_frequency_negative = {}

		word_list = []

		# Iterate thru each of the positive sentences....
		for sentence in positive_sentiments_sentences:
			# tokenize the sentences....
			tokenized_words = word_tokenize(sentence)

			for word in tokenized_words:
				word_list.append(word)

		# generate a unique frequency of words....
		vocab_frequency_positive = dict(Counter(word_list))

		word_list = []

		# Iterate thru each of the negative sentences....
		for sentence in negative_sentiments_sentences:
			# tokenize the sentences....
			tokenized_words = word_tokenize(sentence)

			for word in tokenized_words:
				word_list.append(word)

		# generate a unique frequency of words....
		vocab_frequency_negative = dict(Counter(word_list))


		print("The vocab frequency for positive tweet is : ")
		print(vocab_frequency_positive)

		print("The vocab frequency for negative tweet is : ")
		print(vocab_frequency_negative)

		return vocab_frequency_positive, vocab_frequency_negative




file_name = "D://Coursera//Natural Language Processing Specialization//Course 1 - Classification With Vector Spaces//Week 1//Small_Corpus.xlsx"
sheet_name = "Sheet1"

pre_process = Preprocess_Data()

# extract the corpus list and sentiment list....
positive_sentiments_sentences, positive_sentiments_values, negative_sentiments_sentences, negative_sentiments_values = pre_process.extract_corpus(file_name, sheet_name)

# generate unique vocab list...
vocab_frequency_positive, vocab_frequency_negative = pre_process.generate_unique_vocabulary(positive_sentiments_sentences, negative_sentiments_sentences)




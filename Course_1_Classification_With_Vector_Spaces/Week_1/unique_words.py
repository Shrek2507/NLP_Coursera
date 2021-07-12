import nltk
from nltk.tokenize import word_tokenize

# function to generate unique words...
def generate_unique_words(corpus_list):
	# go thru all the sentences in the corpus list....

	unique_words = list()

	for sentence in corpus_list:
		tokenized_words = word_tokenize(sentence)

		for word in tokenized_words:
			if word not in unique_words:
				unique_words.append(word)
			else:
				pass

	return unique_words

corpus_list = ["I am happy because I am learning NLP", "I hated that movie", "I love working at DL"]

unique_words = generate_unique_words(corpus_list)

print("Total Unique Words/Features in the list are : {0}".format(len(unique_words)))

print()

print("Unique Words in the list are : ")
print(unique_words)

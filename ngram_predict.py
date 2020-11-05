import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import json
import io
from nltk.stem import WordNetLemmatizer
import pickle

english_stopwords = set(stopwords.words('english'))
english_stopwords.remove('not')
lemmatizer = WordNetLemmatizer()

def main():
	english_stopwords = set(stopwords.words('english'))
	english_stopwords.remove('not')
	loaded_vectorizer = pickle.load(open('bigram_vectorizer.pkl', 'rb'))
	loaded_model = pickle.load(open('bigram_model.sav', 'rb'))
	review = input('Enter review to predict sentiment:')
	regex = re.compile(r'[^a-zA-Z\s]')
	review = regex.sub('', review)
	print('Cleaned: ', review)
	filtered = []
	filtered.append(' '.join([word.lower() for word in review.split('\t') if word not in english_stopwords]))
	filtered = loaded_vectorizer.transform(filtered)
	result = loaded_model.predict(filtered)
	if result[0] == 1:
		print('Positive Review')
	elif result[0] == 0:
		print('Negative Review')

if __name__ == "__main__":
	main()

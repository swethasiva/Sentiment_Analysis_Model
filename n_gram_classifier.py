import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import json
import io
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag 

english_stopwords = set(stopwords.words('english'))
english_stopwords.remove('not')
lemmatizer = WordNetLemmatizer()

def load_dataset(filename):
	data = pd.read_csv(filename)
	return data

def clean_data(df):
	reviews = df['review']
	labels = df['sentiment']
	stopword_removed_reviews = []
	lemmatized_reviews = []
	reviews = reviews.replace({'<.*?>': ''}, regex = True)         
	reviews = reviews.replace({'[^A-Za-z]': ' '}, regex = True)
	for review in reviews:
		stopword_removed_reviews.append(' '.join([word.lower() for word in review.split('\t') if word not in english_stopwords]))
	label_encoder = preprocessing.LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	'''for review in stopword_removed_reviews:
		lemmatized_reviews.append(' '.join([lemmatizer.lemmatize(word) for word in review.split('\t')]))'''
	return stopword_removed_reviews, labels

def split_dataset(reviews, labels):
	reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, test_size = 0.2)
	return reviews_train, reviews_test, labels_train, labels_test

def Ngram_Vectorizer(reviews_train, reviews_test):
	ngram_vectorizer = CountVectorizer(analyzer="word", binary=True, ngram_range=(1, 2))
	ngram_vectorizer.fit_transform(reviews_train)
	reviews_train = ngram_vectorizer.transform(reviews_train)
	reviews_test = ngram_vectorizer.transform(reviews_test)
	return reviews_train, reviews_test

def LogisticRegressor(reviews_train, labels_train, reviews_test, labels_test):
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
	    lr = LogisticRegression(C=c)
	    lr.fit(reviews_train, labels_train)
	    print ("Accuracy for C=%s: %s" 
	           % (c, accuracy_score(labels_test, lr.predict(reviews_test))))
    

def main():
	data = load_dataset('IMDB Dataset.csv')
	reviews, labels = clean_data(data)
	print(reviews[3])
	reviews_train, reviews_test, labels_train, labels_test = split_dataset(reviews, labels)
	reviews_train, reviews_test= Ngram_Vectorizer(reviews_train, reviews_test)
	LogisticRegressor(reviews_train, labels_train, reviews_test, labels_test)

if __name__ == "__main__":
	main()
	print('calling')

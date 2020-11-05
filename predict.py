import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import re
import json
import io
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()

def main():
	english_stopwords = set(stopwords.words('english'))
	english_stopwords.remove('not')
	max_len = 100
	loaded_model = load_model('best_model.h5')
	with open('tokenizer.json') as f:
	    data = json.load(f)
	    tokenizer = tokenizer_from_json(data)

	review = input('Enter review to predict sentiment:')
	regex = re.compile(r'[^a-zA-Z\s]')
	review = regex.sub('', review)
	print('Cleaned: ', review)
	words = review.split(' ')
	filtered = [w for w in words if w not in english_stopwords]
	print(filtered)
	flag = [True for word in filtered if word.lower() in ['not', 'no', 'wasn\'t', 'never']]
	if(len(flag)==0):
		flag=[False]
	#filtered = [lemmatizer.lemmatize(w, 'a' if tag[0].lower() == 'j' else tag[0].lower()) for w, tag in pos_tag(filtered) if tag[0].lower() in ['j', 'r', 'n', 'v']]
	#print(filtered)
	filtered = ' '.join(filtered)
	filtered = [filtered.lower()]
	print('Filtered: ', filtered)
	tokenize_words = tokenizer.texts_to_sequences(filtered)
	tokenize_words = pad_sequences(tokenize_words, maxlen=max_len, padding='post', truncating='post')
	print(tokenize_words)
	result = loaded_model.predict(tokenize_words)
	print(result)
	print(flag)
	if result >= 0.8:
	    if (flag[0] == True):
	    	print('Negative')
	    	print("Sentiment Probability \n Positive: " + str(1 - result[0][0]) + ", Negative: " + str(result[0][0]))
	    else:
	    	print('positive')
	    	print("Sentiment Probability \n Positive: " + str(result[0][0]) + ", Negative: " + str(1 - result[0][0]))
	else:
		if(flag[0] == True):
			print('positive')
			print("Sentiment Probability \n Positive: " + str(1 - result[0][0]) + ", Negative: " + str(result[0][0]))
		else:
			print('negative')
			print("Sentiment Probability \n Positive: " + str(result[0][0]) + ", Negative: " + str(1 - result[0][0]))
	return


if __name__ == '__main__':
	main()

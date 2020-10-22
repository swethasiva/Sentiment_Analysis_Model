<<<<<<< HEAD
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
	max_len = 130

	loaded_model = load_model('glove_lstm.h5')
	with open('tokenizer.json') as f:
	    data = json.load(f)
	    tokenizer = tokenizer_from_json(data)
	review = input('Enter review to predict sentiment:')
	regex = re.compile(r'[^a-zA-Z\s]')
	review = regex.sub('', review)
	print('Cleaned: ', review)

	words = review.split(' ')
	filtered = [w for w in words if w not in english_stopwords]
	filtered = ' '.join(filtered)
	filtered = [filtered.lower()]
	filtered = lemmatizer.lemmatize(w, tag[0].lower()) for w, tag in pos_tag(filtered) if tag[0].lower() in ['a', 'r', 'n', 'v']
	print('Filtered: ', filtered)
	tokenize_words = tokenizer.texts_to_sequences(filtered)
	tokenize_words = pad_sequences(tokenize_words, maxlen=max_len, padding='post', truncating='post')
	print(tokenize_words)
	result = loaded_model.predict(tokenize_words)
	print(result)
	return

if __name__ == "__main__":
=======
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
	max_len = 130

	loaded_model = load_model('glove_lstm.h5')
	with open('tokenizer.json') as f:
	    data = json.load(f)
	    tokenizer = tokenizer_from_json(data)
	review = input('Enter review to predict sentiment:')
	regex = re.compile(r'[^a-zA-Z\s]')
	review = regex.sub('', review)
	print('Cleaned: ', review)

	words = review.split(' ')
	filtered = [w for w in words if w not in english_stopwords]
	filtered = ' '.join(filtered)
	filtered = [filtered.lower()]
	filtered = lemmatizer.lemmatize(w, tag[0].lower()) for w, tag in pos_tag(filtered) if tag[0].lower() in ['a', 'r', 'n', 'v']
	print('Filtered: ', filtered)
	tokenize_words = tokenizer.texts_to_sequences(filtered)
	tokenize_words = pad_sequences(tokenize_words, maxlen=max_len, padding='post', truncating='post')
	print(tokenize_words)
	result = loaded_model.predict(tokenize_words)
	print(result)
	return

if __name__ == "__main__":
>>>>>>> 0abb19ce73516a6302f662ed6c3e49de68a55632
	main()
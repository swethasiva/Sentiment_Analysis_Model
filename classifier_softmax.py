<<<<<<< HEAD
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import re
import json
import io
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag 

test_size = 0.2
vocab_size = 40000
trunc_type = 'post'
pad_type = 'post'
embedd_dim = 32
lstm_out = 64
epochs = 15
batch_size = 128
global tokenizer
english_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_dataset(filename):
	data = pd.read_csv(filename)
	#print(data.head)
	return data

def clean_data(df):
	reviews = df['review']
	labels = df['sentiment']
	reviews = reviews.replace({'<.*?>': ''}, regex = True)         
	reviews = reviews.replace({'[^A-Za-z]': ' '}, regex = True)
	print('original length')
	print(reviews[0] )
	reviews = reviews.apply(lambda review : [w for w in review.split() if w not in english_stopwords])
	reviews = reviews.apply(lambda review: [w.lower() for w in review]) 
	print('before lemm')
	print(reviews[0])
	reviews = reviews.apply(lambda review: [lemmatizer.lemmatize(w, tag[0].lower()) for w, tag in pos_tag(review) if tag[0].lower() in ['a', 'r', 'n', 'v']])
	print('after lemm')
	print(reviews[0])
	label_encoder = preprocessing.LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	return reviews, labels

def split_dataset(reviews, labels):
	reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, test_size = 0.2)
	return reviews_train, reviews_test, labels_train, labels_test

def get_max_length(reviews):
	review_length = []
	for review in reviews:
		review_length.append(len(review))
	return int(np.ceil(np.mean(review_length)))

def tokenize_pad_trunc(train_reviews, test_reviews, max_len):
	tokenizer = Tokenizer(num_words = vocab_size, lower= False)
	tokenizer.fit_on_texts(train_reviews)
	print(type(train_reviews))
	train_reviews = tokenizer.texts_to_sequences(train_reviews)
	test_reviews = tokenizer.texts_to_sequences(test_reviews)
	train_reviews = pad_sequences(train_reviews, maxlen=max_len, padding= pad_type,truncating= trunc_type)
	test_reviews = pad_sequences(test_reviews, maxlen=max_len, padding= pad_type ,truncating= trunc_type)
	total_words = int(len(tokenizer.word_index) + 1)

	#print('Encoded X Train\n', train_reviews, '\n')
	#print('Encoded X Test\n', test_reviews, '\n')
	#print('Maximum review length: ', max_len)
	#print(reviews[0])
	return tokenizer, train_reviews, test_reviews, total_words

def save_tokenizer(tokenizers):
	tokenizer_json = tokenizer.to_json()
	with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def sentiment_classification_model(total_words, max_len):
	model = tf.keras.Sequential([
	  tf.keras.layers.Embedding(total_words, embedd_dim, input_length= max_len),
	  tf.keras.layers.LSTM(lstm_out),
	  tf.keras.layers.Dense(2, activation='softmax')])
	model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	#print(model.summary())
	return model

def checkpoint_model():
	checkpoint = ModelCheckpoint(
	  'models/LSTM_softmax.h5',
	  monitor='accuracy',
	  save_best_only=True,
	  verbose=1
	)
	return checkpoint

def plot_training(history):
	import matplotlib.pyplot as plt
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(acc))

	plt.plot(epochs, acc, 'r', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	#plt.savefig('15epochs_lstm_softmax_40000vocab_training_validation_accuracy.png')

	plt.plot(epochs, loss, 'r', label='Training Loss')
	plt.plot(epochs, val_loss, 'b', label='Validation Loss')
	plt.title('Training and validation loss')
	plt.legend()
	#plt.savefig('15epochs_lstm_softmax_40000vocab_training_validation_loss.png')

	plt.show()


data = load_dataset('IMDB Dataset.csv')
reviews, labels = clean_data(data)
reviews_train, reviews_test, labels_train, labels_test = split_dataset(reviews, labels)
max_len = get_max_length(reviews)
tokenizer, reviews_train, reviews_test, total_words = tokenize_pad_trunc(reviews_train, reviews_test, max_len)
save_tokenizer(tokenizer)
model = sentiment_classification_model(total_words, max_len)
checkpoint = checkpoint_model()
history = model.fit(reviews_train, labels_train, validation_data = (reviews_test, labels_test), batch_size = 128, epochs = epochs, callbacks=[checkpoint])
plot_training(history)
#model.save('15_epochs_40000vocab_softmax_lstm.h5')

=======
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import re
import json
import io
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag 

test_size = 0.2
vocab_size = 40000
trunc_type = 'post'
pad_type = 'post'
embedd_dim = 32
lstm_out = 64
epochs = 15
batch_size = 128
global tokenizer
english_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_dataset(filename):
	data = pd.read_csv(filename)
	#print(data.head)
	return data

def clean_data(df):
	reviews = df['review']
	labels = df['sentiment']
	reviews = reviews.replace({'<.*?>': ''}, regex = True)         
	reviews = reviews.replace({'[^A-Za-z]': ' '}, regex = True)
	print('original length')
	print(reviews[0] )
	reviews = reviews.apply(lambda review : [w for w in review.split() if w not in english_stopwords])
	reviews = reviews.apply(lambda review: [w.lower() for w in review]) 
	print('before lemm')
	print(reviews[0])
	reviews = reviews.apply(lambda review: [lemmatizer.lemmatize(w, tag[0].lower()) for w, tag in pos_tag(review) if tag[0].lower() in ['a', 'r', 'n', 'v']])
	print('after lemm')
	print(reviews[0])
	label_encoder = preprocessing.LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	return reviews, labels

def split_dataset(reviews, labels):
	reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, test_size = 0.2)
	return reviews_train, reviews_test, labels_train, labels_test

def get_max_length(reviews):
	review_length = []
	for review in reviews:
		review_length.append(len(review))
	return int(np.ceil(np.mean(review_length)))

def tokenize_pad_trunc(train_reviews, test_reviews, max_len):
	tokenizer = Tokenizer(num_words = vocab_size, lower= False)
	tokenizer.fit_on_texts(train_reviews)
	print(type(train_reviews))
	train_reviews = tokenizer.texts_to_sequences(train_reviews)
	test_reviews = tokenizer.texts_to_sequences(test_reviews)
	train_reviews = pad_sequences(train_reviews, maxlen=max_len, padding= pad_type,truncating= trunc_type)
	test_reviews = pad_sequences(test_reviews, maxlen=max_len, padding= pad_type ,truncating= trunc_type)
	total_words = int(len(tokenizer.word_index) + 1)

	#print('Encoded X Train\n', train_reviews, '\n')
	#print('Encoded X Test\n', test_reviews, '\n')
	#print('Maximum review length: ', max_len)
	#print(reviews[0])
	return tokenizer, train_reviews, test_reviews, total_words

def save_tokenizer(tokenizers):
	tokenizer_json = tokenizer.to_json()
	with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def sentiment_classification_model(total_words, max_len):
	model = tf.keras.Sequential([
	  tf.keras.layers.Embedding(total_words, embedd_dim, input_length= max_len),
	  tf.keras.layers.LSTM(lstm_out),
	  tf.keras.layers.Dense(2, activation='softmax')])
	model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	#print(model.summary())
	return model

def checkpoint_model():
	checkpoint = ModelCheckpoint(
	  'models/LSTM_softmax.h5',
	  monitor='accuracy',
	  save_best_only=True,
	  verbose=1
	)
	return checkpoint

def plot_training(history):
	import matplotlib.pyplot as plt
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(acc))

	plt.plot(epochs, acc, 'r', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	#plt.savefig('15epochs_lstm_softmax_40000vocab_training_validation_accuracy.png')

	plt.plot(epochs, loss, 'r', label='Training Loss')
	plt.plot(epochs, val_loss, 'b', label='Validation Loss')
	plt.title('Training and validation loss')
	plt.legend()
	#plt.savefig('15epochs_lstm_softmax_40000vocab_training_validation_loss.png')

	plt.show()


data = load_dataset('IMDB Dataset.csv')
reviews, labels = clean_data(data)
reviews_train, reviews_test, labels_train, labels_test = split_dataset(reviews, labels)
max_len = get_max_length(reviews)
tokenizer, reviews_train, reviews_test, total_words = tokenize_pad_trunc(reviews_train, reviews_test, max_len)
save_tokenizer(tokenizer)
model = sentiment_classification_model(total_words, max_len)
checkpoint = checkpoint_model()
history = model.fit(reviews_train, labels_train, validation_data = (reviews_test, labels_test), batch_size = 128, epochs = epochs, callbacks=[checkpoint])
plot_training(history)
#model.save('15_epochs_40000vocab_softmax_lstm.h5')

>>>>>>> 0abb19ce73516a6302f662ed6c3e49de68a55632

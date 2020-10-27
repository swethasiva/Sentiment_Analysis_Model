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

test_size = 0.2
vocab_size = 40000
trunc_type = 'post'
pad_type = 'post'
embedd_dim = 100
lstm_out = 64
epochs = 5
batch_size = 128
embedding_dim = 100 
global tokenizer
english_stopwords = set(stopwords.words('english'))
english_stopwords.remove('not')
embeddings_index = {}

def load_dataset(filename):
	data = pd.read_csv(filename)
	#print(data.head)
	return data

def clean_data(df):
	reviews = df['review']
	labels = df['sentiment']
	reviews = reviews.replace({'<.*?>': ''}, regex = True)         
	reviews = reviews.replace({'[^A-Za-z]': ' '}, regex = True)
	#print(len(reviews[0]))
	reviews = reviews.apply(lambda review : [w for w in review.split() if w not in english_stopwords])
	#print(len(reviews[0]))
	reviews = reviews.apply(lambda review: [w.lower() for w in review]) 
	#print(labels.head())
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

def load_glove():
	f = open("glove.6B.100d.txt", encoding='utf-8') #added , encoding='utf-8'
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype="float32")
		embeddings_index[word] = coefs
	f.close()
	print("found %s word vectors." % len (embeddings_index))
	return

def embedd_matrix():
	embedding_matrix = np.zeros((total_words, embedding_dim)) 
	for word, i in word_index.items():
	    if i < total_words:
	        embedding_vector = embeddings_index.get(word) 
	        if embedding_vector is not None:
	            embedding_matrix[i] = embedding_vector 
	print(embedding_matrix.shape)
	return embedding_matrix

def sentiment_classification_model(total_words, max_len, embedding_matrix):
	model = tf.keras.Sequential([
	  tf.keras.layers.Embedding(total_words, embedd_dim, input_length= max_len),
	  tf.keras.layers.LSTM(lstm_out),
	  tf.keras.layers.Dense(1, activation='sigmoid')])
	model.layers[0].set_weights([embedding_matrix])
	model.layers[0].trainable = False
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
	plt.savefig('glove_lstm_sigmoid_training_validation_accuracy.png')

	plt.plot(epochs, loss, 'r', label='Training Loss')
	plt.plot(epochs, val_loss, 'b', label='Validation Loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig('glove_lstm_sigmoid_training_validation_loss.png')

	plt.show()

data = load_dataset('IMDB Dataset.csv')
reviews, labels = clean_data(data)
reviews_train, reviews_test, labels_train, labels_test = split_dataset(reviews, labels)
#max_len = get_max_length(reviews)
max_len = 100
tokenizer, reviews_train, reviews_test, total_words = tokenize_pad_trunc(reviews_train, reviews_test, max_len)
word_index = tokenizer.word_index
load_glove()
embedding_matrix = embedd_matrix()
model = sentiment_classification_model(total_words, max_len, embedding_matrix)
checkpoint = checkpoint_model()
history = model.fit(reviews_train, labels_train, validation_data = (reviews_test, labels_test), batch_size = 128, epochs = 5, callbacks=[checkpoint])
plot_training(history)
model.save('glove_sigmoid_lstm.h5')


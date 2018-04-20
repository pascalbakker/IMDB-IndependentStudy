import pandas
import os
import keras
import zipfile
import requests
import tqdm
import numpy as np
from sklearn import dummy, metrics, cross_validation, ensemble
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras import preprocessing
from keras.layers import Embedding
from keras.layers import Flatten, Dense
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers import SimpleRNN
from keras.preprocessing import sequence
from keras.layers import Dense, Activation
from keras.layers import Input, LSTM, Dense
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#Get data and split it
imdbdir = '/Users/pascal/Downloads/aclImdb'
traindir = os.path.join(imdbdir,'train')


maxlen = 100
training_samples = 18000
validation_samples = 7000
max_words = 10000

def getData():
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(traindir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return labels,texts


def loadData(labels,texts):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    data = pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]
    return (x_train,y_train),(x_val,y_val)


def MainProgram():
    #Get data
    labels, texts = getData()
    (x_train, y_train), (x_val, y_val) = loadData(labels, texts)


    glove_dir = '/Users/pascal/Downloads/glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    #Embedding
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    #Define model
    dout = 0.5



    # Create input layer

    #%%
    model = Sequential()

    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    #model.add(Flatten())
    model.add(LSTM(embedding_dim))
    """
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dout))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dout))
    """
    #model.add(Dense(50, activation='sigmoid'))

    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=30,
                        batch_size=128,
                        validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')


    #Display data
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def splitBoW():
    glove_dir = '/Users/pascal/Downloads/aclImdb'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()


#Main program
MainProgram()


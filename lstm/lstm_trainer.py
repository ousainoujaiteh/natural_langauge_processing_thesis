'''
    Sequence classification is a predictive modeling problem where you have some sequence of inputs over space or
    time and the task is to predict a category for the sequence.

    What makes this problem difficult is that the sequences can vary in length, be comprised of a very large vocabulary
    of input symbols and may require the model to learn the long-term context or dependencies between symbols in the input sequence.
'''
import time
import numpy

import matplotlib.pyplot as plt

# import tensorflow as tf
# import keras

# import for keras modules
from keras.layers import LSTM
from keras.layers import Dense
from keras.datasets import imdb
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

# fix random seed for reproducibility
numpy.random.seed(7)

'''
    load the IMDB dataset. We are constraining the dataset to the top 5,000 words.
    And also split the dataset into train (50%) and test (50%) sets.
'''
# How many unique words I want to load into training and testing dataset
top_words = 5000
(X_train,y_train) , (X_test,y_test) = imdb.load_data(num_words=top_words)

word_index = imdb.get_word_index()

print('# of training samples: {}'.format(len(X_train)))
print('# of testing samples: {}'.format(len(y_train)))


'''
    Decode the word index from integer values to letters 
'''
def decode_review(text):
    index_to_word = {}
    for key,value in word_index.items():
        index_to_word[value] = key

    return ''.join([index_to_word[x] for x in text])


'''
    Now  to truncate and pad the input sequences so that they are all the same length for modeling. 
    The model will learn the zero values carry no information so indeed the sequences are not the 
    same length in terms of content, but same length vectors is required to perform the computation in Keras.    
    backpropagation
    trim them to their first 500 words
'''
max_review_length = 500
X_train = sequence.pad_sequences(X_train,maxlen=max_review_length)
X_test  = sequence.pad_sequences(X_test,maxlen=max_review_length)


'''
    Now to define,compile and fit the LSTM model
    Embedding layer : This is the step that converts the input data into dense vectors of fixed
    size that better suit the neural network
    The first layer is the Embedded layer that uses 32 length vectors to represent each word.
    The next layer is the LSTM layer with 100 memory units (smart neurons).
    Finally, because this is a classification problem we use a Dense output layer with a single neuron and 
    a sigmoid activation function to make 0 or 1 predictions for the two classes (positive and negative) in the problem.
    
    Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). 
    The efficient ADAM optimization algorithm is used. The model is fit for only 2 epochs because it quickly overfits 
    the problem. A large batch size of 64 reviews is used to space out weight updates.
'''
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words,embedding_vector_length,input_length=max_review_length))
#model.add(Dropout(0.2))
model.add(LSTM(100))
#model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

# Log the model for tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=64,callbacks=[tensorboard])

history_dict = history.history
history_dict.keys()


acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''
    Once fit, now estimate the performance of unseen reviews
'''
scores = model.evaluate(X_test,y_test,verbose=0)
print("Test Loss: %.2f%%" % (scores[0]*100))
print("Test Accuracy: %.2f%%" % (scores[1]*100))


print("Test set for the predictions",decode_review(X_test[0]))
'''
    Get Predictions of the model
'''
predictions = model.predict(X_test, batch_size=64)


for num in predictions:
    print("Predicted Value",num)

'''
    Save the model weights
'''
model_path = '/home/ousainou/PycharmProjects/Thesis/data/models/imdb'
model.save(model_path + 'lstm_imdb_model.h5')
model.save_weights(model_path + 'lstm_imdb_weights.h5')


'''
                        APPLYING DROPOUT
                        
    Recurrent Neural networks like LSTM generally have the problem of overfitting.

    Dropout can be applied between layers using the Dropout Keras layer. 
    We can do this easily by adding new Dropout layers between the Embedding and 
    LSTM layers and the LSTM and Dense output layers.
'''
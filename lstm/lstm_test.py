# the numpy array
import numpy as np


# import for keras modules
from keras.datasets import imdb
from keras.models import load_model
from keras.preprocessing import sequence ,text

# load the save model
model_path = '/home/ousainou/PycharmProjects/Thesis/data/models/imdb'
model = load_model(model_path + 'lstm_imdb_model.h5')
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


def decode_sentence(sentence):
    word_index = imdb.get_word_index()
    '''
         It is converting a text into sequence of words or token
    '''
    sentence = text.text_to_word_sequence(sentence,filters='!”#$%&()*+,-./:;?@[\\]^_`{|}~\t\n',lower=True)
    '''
                            SEEN FROM THE INTERNET
        Converitng the word list into numpy array of word indexes , with 0 for unknown words
        for each string in the data file.
    '''
    sentence = np.array([word_index[word] if word in word_index else 0 for word in sentence])
    print("sentence",sentence)
    sentence[sentence > 5000] = 2
    l = 500 - len(sentence)
    sentence = np.pad(sentence,(l,0),'constant')
    sentence = sentence.reshape(1,-1)
    return sentence

result = decode_sentence("I do not like this")

predictions = model.predict(result)

def truncate(n):
    return int(n * 1000) / 1000

for pred in predictions:
    value = round(truncate(pred))
    if value == 0:
        print("Negative Comment")
    else:
        print("Positive Comment")


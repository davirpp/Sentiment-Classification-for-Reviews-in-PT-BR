import json
import re
import tensorflow as tf
import warnings
from statistics import mode

from nltk.tokenize import word_tokenize
from unidecode import unidecode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, 
    Dense, Dropout, Embedding, 
    GlobalAvgPool1D, LSTM
)

# Have to install the tokenizer from nltk

warnings.filterwarnings('ignore')

f = open('tokenizer_config.json')
tokenizer_config = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_config)

# String preprocessing, embedding and padding
def preprocess(text):
    input_text = text.lower()
    input_text = unidecode(input_text)
    input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)
    input_text = word_tokenize(input_text)
    input_text = ' '.join(input_text)
    input_text = tokenizer.texts_to_sequences([input_text])
    input_text = pad_sequences(input_text, maxlen=500, padding='pre')
    return input_text

# ---------------- LOAD MODELS ----------------
models = []

# MLP1
mlp1 = Sequential()
mlp1.add(Embedding(10000, 16, input_length=500))
mlp1.add(Dropout(0.2))
mlp1.add(GlobalAvgPool1D())
mlp1.add(Dropout(0.4))
mlp1.add(Dense(1, activation='sigmoid'))
mlp1.load_weights('MLP/weights_mlp.h5')
models.append(mlp1)
# ---------------------------------------------

# 2 LSTM v1
bi_lstm = Sequential()
bi_lstm.add(Embedding(10000, 64))
bi_lstm.add(Bidirectional(LSTM(100, return_sequences=True)))
bi_lstm.add(BatchNormalization())
bi_lstm.add(Bidirectional(LSTM(32)))
bi_lstm.add(Dropout(0.1))
bi_lstm.add(Dense(256, activation="relu"))
bi_lstm.add(Dropout(0.2))
bi_lstm.add(Dense(1, activation="sigmoid"))
bi_lstm.load_weights('BILSTM/weights_BILSTM_v1.h5')
models.append(bi_lstm)
# ---------------------------------------------

# 2 LSTM v2
bi_lstm2 = Sequential()
bi_lstm2.add(Embedding(10000, 64))
bi_lstm2.add(Bidirectional(LSTM(32, return_sequences=True)))
bi_lstm2.add(BatchNormalization())
bi_lstm2.add(Bidirectional(LSTM(64, dropout=0.1)))
bi_lstm2.add(Dense(256, activation="relu"))
bi_lstm2.add(Dropout(0.2))
bi_lstm2.add(Dense(1, activation="sigmoid"))
bi_lstm2.load_weights('BILSTM/weights_BILSTM_v2.h5')
models.append(bi_lstm2)
# ---------------------------------------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Speed up the prediction
@tf.function
def predict_(model, input_data):
    return model(input_data, training=False)

# Logic part to do the ensemble and increase accuracy
def ensemble(text):
    input_text = preprocess(text)
    preds = [predict_(model, input_text) for model in models]
    predictions = [1 if p >= 0.5 else 0 for p in preds]
    predict = mode(predictions)
    print('preds:', preds, '\npredictions:', predictions)
    print(f'Prediction: {predict}')

def main():
    text = input('\nEnter text: ')
    while text != 'exit':
        ensemble(text)
        text = input('\nEnter text: ')

if __name__ == '__main__':
    main()

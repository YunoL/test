from flask import Flask

from keras.models import Sequential
from keras import layers
import pickle
from flask import request
from keras.utils import pad_sequences
import numpy as np


tokenzier_file = open("tokenizer.pkl",'rb')
tokenizer = pickle.load(tokenzier_file)
vocab_size = len(tokenizer.word_index) + 1

label_encoder_file = open("label_encoder.pkl",'rb')
label_encoder = pickle.load(label_encoder_file)

embedding_dim = 100
def create_model(dropout=0):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                               output_dim=embedding_dim, 
                               input_length=1009))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(293, activation='softmax'))
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    return model


app = Flask(__name__)

model = create_model()
model.load_weights("/home/yuno/Getpro/labtech/xray_class_title_my_model.best.hdf5")



def predict(titles):
    print(titles)
    global tokenzer
    x = tokenizer.texts_to_sequences([titles])
    x = pad_sequences(x, padding='post', maxlen=1009)

    print(x)
    pred = model.predict(x)
    pred = np.argmax(pred)

    res = label_encoder.inverse_transform([pred])[0]

    return res

@app.route("/", methods=['GET'])
def hello_world():
    titles = request.args["titles"]
    return predict(titles)
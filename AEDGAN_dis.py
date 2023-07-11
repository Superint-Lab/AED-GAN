import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras.backend as K

use_cell = 1
rnn_layer_discriminator = True 
disc_input_mul = 2

def build_discriminator(self):
    model = Sequential()
    model.add(Reshape((disc_input_mul*(self.data_rows + self.target_data_rows), self.data_cols), input_shape=self.combined_shape))

    if rnn_layer_discriminator:
        if use_cell == 1:
            model.add(GRU(256, return_sequences=True, kernel_initializer="normal"))
        if use_cell == 2:
            model.add(LSTM(256, return_sequences=True, kernel_initializer="normal"))

    model.add(Dense(512, activation="tanh", kernel_initializer="normal"))
    model.add(Dense(256, activation="tanh", kernel_initializer="normal"))
    model.add(Dense(128, activation="tanh", kernel_initializer="normal"))
    model.add(Dense(1, activation="tanh", kernel_initializer="normal"))
    model.summary()


    steps = Input(shape=self.combined_shape)
    layer1 = Reshape((disc_input_mul*(self.data_rows + self.target_data_rows), self.data_cols), input_shape=self.combined_shape)(steps)

    if rnn_layer_discriminator:
        if use_cell == 1:
            layer2 = GRU(256, return_sequences=True, kernel_initializer="normal")(layer1)
        if use_cell == 2:
            layer2 = LSTM(256, return_sequences=True, kernel_initializer="normal")(layer1)

        layer3 = Dense(512, activation="tanh", kernel_initializer="normal")(layer2)
    else:
        layer3 = Dense(512, activation="tanh", kernel_initializer="normal")(layer1)

    layer4 = Dense(256, activation="tanh", kernel_initializer="normal")(layer3)
    layer5 = Dense(128, activation="tanh", kernel_initializer="normal")(layer4)
    validity = Dense(1, activation="sigmoid", kernel_initializer="normal")(layer5)

    return Model(steps, [validity, layer5])
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras.backend as K


batch_size_global = 32
latent_dim = 256 
use_cell = 1
set_attention = True

def build_generator(self):
    encoder_inputs = Input(shape=(self.data_rows, self.data_cols), name='encoder_inputs')

    if use_cell == 1:
        encoder_gru = Bidirectional(GRU(latent_dim, kernel_initializer="normal", return_sequences=True, return_state=True, name='encoder_gru'))
        encoder_outputs, forward_state_h, backward_state_h = encoder_gru(encoder_inputs)
        state_h = Concatenate()([forward_state_h, backward_state_h])
        states = [forward_state_h, backward_state_h]
        attention = BahdanauAttention(latent_dim, verbose=0)
        decoder_inputs = Input(shape=(None, self.poa_num), name='decoder_inputs') 
        decoder_gru = Bidirectional(GRU(latent_dim, return_state=True, name='decoder_gru'))
        decoder_dense = Dense(self.poa_num, activation='softmax',  name='decoder_dense')

    if use_cell == 2:
        encoder_lstm = Bidirectional(LSTM(latent_dim, kernel_initializer="normal", return_sequences=True, return_state=True, name='encoder_lstm'))
        encoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = encoder_lstm(encoder_inputs)
        state_h = Concatenate()([forward_state_h, backward_state_h])
        states = [forward_state_h, forward_state_c, backward_state_h, backward_state_c]
        attention = BahdanauAttention(latent_dim, verbose=0)
        decoder_inputs = Input(shape=(None, self.poa_num), name='decoder_inputs')
        decoder_lstm = Bidirectional(LSTM(latent_dim, return_state=True, name='decoder_lstm'))
        decoder_dense = Dense(self.poa_num, activation='softmax',  name='decoder_dense')


    all_outputs = []
    batch_size = batch_size_global 
    decoder_outputs = state_h 
    states = states 

    for loop_num in range(self.target_data_rows):

        inputs = decoder_inputs[:, loop_num] 
        inputs = tf.expand_dims(inputs, axis=1) 

        if set_attention:
            # attention
            context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)
            context_vector = tf.expand_dims(context_vector, 1)

            # concatenate input + context vector to find next decoder's input
            inputs = tf.cast(inputs, tf.float32)
            inputs = tf.concat([context_vector, inputs], axis=-1)
        else:
            inputs = tf.cast(inputs, tf.float32)


        # Run the decoder on one time step
        if use_cell == 1:
            decoder_outputs, forward_state_h, backward_state_h = decoder_gru(inputs, initial_state=states)
        if use_cell == 2:
            decoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = decoder_lstm(inputs, initial_state=states)

        outputs = decoder_dense(decoder_outputs)
        outputs = tf.expand_dims(outputs, 1) 

        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)

        # update the states
        if use_cell == 1:
            states = [forward_state_h, backward_state_h]
        if use_cell == 2:
            states = [forward_state_h, forward_state_c, backward_state_h, backward_state_c]

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='model_encoder_decoder')
    model.summary()
    return model
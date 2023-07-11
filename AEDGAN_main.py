# AEDGAN for PMD

from __future__ import print_function, division
import tensorflow as tf
 

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


import csv
import glob
import numpy as np
import time
import math
import random
import warnings
from keras.layers import *
import keras.backend
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.models import *
from keras.utils import *
from keras import optimizers
import keras.backend as K
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker
from accuracy_funct import accuracy_ED
from preprocess_funct import preprocess_data1

AEDGAN_folder_path = './Jupyter/AEDGAN/'
data_source_path = './Data/csv_files(raw_data)/new/*.csv' #Folder with raw data csv

processed_data_path = AEDGAN_folder_path + 'Processed_data/'
log_path = AEDGAN_folder_path + 'Log/AEDGAN_TEST(ablation)/'

csv_data_path = processed_data_path
logging_path = log_path

print(csv_data_path)
print(logging_path)

################################################# parameters ##################################################

poa_num = 12 
vector_dim = 12 
batch_size_global = 32
epochs = 5000 
total_size = 1  
train_size = 0.8  
data_shuffle_epoch = 3  
fact_matching_cnt = 1 
use_gan = True 
latent_dim = 256
disc_train_num = 2 
disc_input_mul = 2 
label_smoothing_valid = 0 
label_smoothing_fake = 0 
set_attention = True 
use_cell = 1 
rnn_layer_discriminator = True 

random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

############################################### parameters #######################################################


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units, verbose=0):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    self.verbose= verbose

  def call(self, query, values):
    if self.verbose:
      print('\n******* Bahdanau Attention STARTS******')
      print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
      print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)
    
    if self.verbose:
      print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    if self.verbose:
      print('score: (batch_size, max_length, 1) ',score.shape)
    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)
    if self.verbose:
      print('attention_weights: (batch_size, max_length, 1) ',attention_weights.shape)
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    if self.verbose:
      print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ',context_vector.shape)
    context_vector = tf.reduce_sum(context_vector, axis=1)
    if self.verbose:
      print('context_vector after reduce_sum: (batch_size, hidden_size) ',context_vector.shape)
      print('\n******* Bahdanau Attention ENDS******')
    return context_vector, attention_weights


class AEDGAN:
    def __init__(self):
        self.data_rows = seq_len #22
        self.data_cols = vector_dim #12
        self.target_data_rows = target_len #7
        self.channels = 1
        self.poa_num = poa_num


        self.input_shape = (self.data_rows, self.data_cols, self.channels) #(22, 12, 1)
        self.output_shape = (self.target_data_rows, self.data_cols, self.channels) #(7, 12, 1)
        self.combined_shape = (disc_input_mul*(self.data_rows + self.target_data_rows), self.data_cols, self.channels) #(29, 12, 1)

        self.decoder_input_shape = (self.target_data_rows, self.data_cols, self.channels) #(_, 7, 12)

        # Optimizer, Loss function
        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.5)
        loss_function = "binary_crossentropy"
        loss_function1 = "categorical_crossentropy"
        loss_function2 = "mse"

        # Use Fact matching to train Generator
        self.generator = self.build_generator()
        self.generator.compile(loss=loss_function1, optimizer=optimizer, metrics=["acc"])
        
        #  Train discriminator based on output of generator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

        # previous steps shape = input_shape
        previous_steps = Input(shape=self.input_shape)
        decoder_inputs = Input(shape=self.decoder_input_shape)

        # generator takes previous_steps as input, then generated next multiple steps(gen_next_step)
        gen_next_step = self.generator([previous_steps,decoder_inputs]) #changed
        gen_next_step = tf.expand_dims(gen_next_step, axis=3)


        # merged_data combined previous steps and generated steps, and becomes discriminator's fake input
        merged_data = concatenate([previous_steps, gen_next_step], axis=1)
        merged_data_dou = concatenate(disc_input_mul*[merged_data], axis=1)

        validity,feature_validity = self.discriminator(merged_data_dou)

        # Do not update discriminator when update/train generator
        self.discriminator.trainable = False

        self.combined = Model([previous_steps,decoder_inputs], feature_validity) #changed
        self.combined.compile(loss='mse', optimizer=optimizer, metrics='mse')



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

    def train(self):

        train_data, test_data = preprocess_data1(seq_len,target_len)

        batch_size = batch_size_global

        #test data processing can come here
        test_size = len(test_data) 
        test_data = test_data[0:test_size]
        test_prev_steps = test_data[:, 0:self.data_rows] # data_rows = seq_len = 22
        test_real_next_step = test_data[:, self.data_rows:]
        test_real_next_step = np.reshape(
            test_real_next_step,
            (len(test_real_next_step), self.target_data_rows, self.data_cols, self.channels),
        )


        valid = np.ones((batch_size, disc_input_mul*(seq_len+target_len), 1)) - label_smoothing_valid # 0.9
        fake = np.zeros((batch_size, disc_input_mul*(seq_len+target_len), 1)) + label_smoothing_fake # 0.1 

        ######################################### Training ##############################################################

        start = time.time()
        training_accumulated_time = 0
        accuracy_list = []
        step_acc_list = []

        for epoch in range(epochs+1):

            np.random.shuffle(train_data)

            idx = np.random.randint(0, len(train_data), batch_size)
            real_combined_steps = train_data[idx] 

            # real_combined_steps divided into previous steps and real_next_step
            prev_steps = real_combined_steps[:, 0:self.data_rows] # data_rows = seq_len = 22
            real_next_step = real_combined_steps[:, self.data_rows:]
       
            real_next_step = np.reshape(real_next_step, (batch_size, self.target_data_rows, self.data_cols, self.channels))

            #decoder input for teacher forcing
            real_next_step_sequeeze = tf.squeeze(real_next_step)
            start_token = np.zeros((batch_size, 1, vector_dim))
            decoder_input = np.concatenate((start_token, real_next_step_sequeeze[:, :-1]), axis=1)   


            # generator predict next steps using pre_steps
            gen_next_step = self.generator.predict([prev_steps, decoder_input]) #changed
            gen_next_step = tf.expand_dims(gen_next_step, axis=3)

            # fake combined steps
            gen_combined_steps = np.concatenate((prev_steps, gen_next_step), axis=1)
            
            if use_gan:
                # multiply the inputs of discriminator
                real_combined_steps_dou = concatenate(disc_input_mul*[real_combined_steps], axis=1)
                gen_combined_steps_dou = concatenate(disc_input_mul*[gen_combined_steps], axis=1)


                for k in range(disc_train_num):
                    d_loss_real = self.discriminator.train_on_batch(
                        real_combined_steps_dou, valid
                    )
                    d_loss_fake = self.discriminator.train_on_batch(
                        gen_combined_steps_dou, fake
                    )
                    
                    d_out, feature_d = self.discriminator.predict(real_combined_steps_dou)

                g_loss = self.combined.train_on_batch([prev_steps, decoder_input], feature_d) 
                

            # Fact Matching
            # trains to reduce the difference between the next step generated by the generator and the actual next step.
            y1 = tf.squeeze(prev_steps)
            y2 = tf.squeeze(real_next_step)
            prev_steps = y1
            real_next_step = y2

            
            for fact_matching in range(fact_matching_cnt):
                g_loss_FM = self.generator.train_on_batch([prev_steps, decoder_input], real_next_step) 
            
            if epoch % 100 == 0:
                spending_time = time.time()-start
                training_accumulated_time +=spending_time
                # print("epoch: "+ str(epoch)+ "d_loss: "+ str(d_loss_real + d_loss_fake)+ "g_loss_FM: ",
                #     # + str(g_loss + g_loss_FM)   
                # )
                testing_start = time.time()
                current_test_acc, current_step_acc = self.testing(epoch, test_data, training_accumulated_time, self.generator, test_prev_steps, test_real_next_step, test_size)
                accuracy_list.append(current_test_acc) #final acc
                step_acc_list.append(current_step_acc) #acc of each step
                testing_spend = time.time()-testing_start
                start=time.time()

            # if epoch == (epochs):
            #     self.logging(str(accuracy_list))
            #     self.logging(str(accuracy_list.index(max(accuracy_list))))
            #     self.logging(str(max(accuracy_list)))
            #     self.logging(str(step_acc_list[accuracy_list.index(max(accuracy_list))]))
            #     self.logging('Training time:')
            #     self.logging(str(training_accumulated_time))
            #     self.logging('Testing time:')
            #     self.logging(str(testing_spend))
            #     self.plot_acc_graph(accuracy_list)

            if epoch == (epochs):
                self.logging('Exp_'+ str(expnum))
                for log_len in range(target_len):
                    acc_item = step_acc_list[accuracy_list.index(max(accuracy_list))][log_len]
                    self.logging(str(acc_item))
                self.logging('')

                
                
                
        ############################################# Training end ###########################################################

        ############################################## Testing ###########################################################

    def testing(self, epoch, test_data, time_, generator, test_prev_steps, test_real_next_step, test_size):

        def generator_test(test_prev_steps_temp):

            if use_cell == 1:
                if set_attention:
                    encoder_gru = generator.layers[1] 
                    attention = generator.layers[5] 
                    decoder_gru = generator.layers[10] 
                    decoder_dense = generator.layers[10+5*(target_len-1)+1]
                    # print("encoder_gru:", encoder_gru)
                    # print("decoder_gru:", decoder_gru)
                else:
                    encoder_gru = generator.layers[6]
                    decoder_gru = generator.layers[9]
                    decoder_dense = generator.layers[9+3*(target_len-2)+1]
                    # print("encoder_gru_noATT:", encoder_gru)
                    # print("decoder_gru_noATT:", decoder_gru)

                encoder_inputs = test_prev_steps_temp
                encoder_outputs, forward_state_h, backward_state_h = encoder_gru(encoder_inputs)

                state_h = Concatenate()([forward_state_h, backward_state_h])
                states = [forward_state_h, backward_state_h]

            if use_cell == 2:
                if set_attention:
                    encoder_lstm = generator.layers[1] 
                    attention = generator.layers[5] 
                    decoder_lstm = generator.layers[10] 
                    decoder_dense = generator.layers[10+5*(target_len-1)+1]
                    # print("encoder_lstm", encoder_lstm)
                    # print("decoder_lstm", decoder_lstm)
                else:
                    encoder_lstm = generator.layers[6]
                    decoder_lstm = generator.layers[9]
                    decoder_dense = generator.layers[9+3*(target_len-2)+1]
                    # print("encoder_lstm_Xatt:", encoder_lstm)
                    # print("decoder_lstm_Xatt:", decoder_lstm)

                encoder_inputs = test_prev_steps_temp
                encoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = encoder_lstm(encoder_inputs)

                state_h = Concatenate()([forward_state_h, backward_state_h])
                states = [forward_state_h, forward_state_c, backward_state_h, backward_state_c]


            all_outputs = []
            # decoder_input_data = np.zeros((test_size, 1, poa_num)) 
            decoder_input_data = np.zeros((slide_win, 1, poa_num)) 
            inputs = decoder_input_data

            decoder_outputs = state_h 
            states = states 

            # repeat target lenght times
            for _ in range(target_len):

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
                    decoder_outputs, forward_state_h, backward_state_h= decoder_gru(inputs, initial_state=states)
                if use_cell == 2:
                    decoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = decoder_lstm(inputs, initial_state=states)

                outputs = decoder_dense(decoder_outputs)
                outputs = tf.expand_dims(outputs, 1) 


                # Store the current prediction 
                all_outputs.append(outputs)
                # Reinject the outputs as inputs for the next loop iteration
                # as well as update the states
                inputs = outputs

                if use_cell == 1:
                    states = [forward_state_h, backward_state_h]
                if use_cell == 2:
                    states = [forward_state_h, forward_state_c, backward_state_h, backward_state_c]

            decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

            return decoder_outputs


        #slide_win = 500 
        slide_win = len(test_data) #for Pangyo can just use len of test data, but suwon dataset need to divide
        
        test_gen_next_step = []
        for i in range(int(len(test_data)/slide_win)):
            # print(i)
            test_prev_steps_temp = test_prev_steps[i*slide_win: (i+1)*slide_win]
            test_prev_steps_temp = tf.squeeze(test_prev_steps_temp)
            #print(test_prev_steps_temp.shape)
            test_gen_next_step_temp = generator_test(test_prev_steps_temp)
            test_gen_next_step.append(test_gen_next_step_temp)
            # print(np.shape(test_gen_next_step))

        # print(np.shape(test_gen_next_step))

        test_gen_next_step = np.reshape(test_gen_next_step, ((i+1)*slide_win, target_len, vector_dim))
        test_gen_next_step = tf.expand_dims(test_gen_next_step, axis=3)

        test_real_next_step_slide = test_real_next_step[:(i+1)*slide_win]

        # print("test_real_next_step_slide", test_real_next_step_slide.shape)
        # print("test_gen_next_step", test_gen_next_step.shape)

        #testing_acc = 100 * accuracy_ED(test_real_next_step_slide, test_gen_next_step)
        testing_acc, step_acc = accuracy_ED(test_real_next_step_slide, test_gen_next_step)


        print(
            "%d [epoch: %d] [test acc.: %.2f%%] %.2f"
            % (
                seq_len,
                epoch,
                testing_acc,
                time_
            )
        )

        # # Save model if neccessary
        # save_model(model=self.generator, filepath= logging_path + "SavedModel_" 
        #        + modeltype + expnum + str(seq_len) + '_' + str(target_len))

        return testing_acc, step_acc


            
    def logging(self,txt):
        log_file = open(logging_path + modeltype + str(seq_len) + '_' + str(target_len)+'.txt', 'a')
        log_file.write(txt+'\n')
        log_file.close()
        

    def plot_acc_graph(self,accuracy_list):
        y_axis_value = accuracy_list
        x_axis_value = []
        i = 0
        while i <= epochs:
            x_axis_value.append(i)
            i += 100
        x = np.array(x_axis_value)
        y = np.array(accuracy_list)


        fig, ax = plt.subplots()
        ax.plot(x,y)

        def annot_max(x,y, ax=None):
            xmax = x[np.argmax(y)]
            ymax = y.max()
            text= "Epoch={}, Acc={:.2f}%".format(xmax, ymax)
            if not ax:
                ax=plt.gca()
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            kw = dict(xycoords='data',textcoords="axes fraction",
                        bbox=bbox_props, ha="right", va="top")
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

        annot_max(x,y)
        plt.savefig(logging_path +
                    modeltype +  expnum +  str(seq_len) + '_' + str(target_len)+'.png',
                    dpi=500, transparent=False)
    

        ############################################# Testing end #########################################################



if __name__ == "__main__":

    batch_size_list = [32]
    disc_input_mul_list = [2]

    input_len_list = [24]
    output_len_list = [9]

    for k in batch_size_list:
        batch_size_global = k
        for l in disc_input_mul_list:
            disc_input_mul = l
            for j in range(len(output_len_list)):
                for p in range(1, 31):
                    for i in range(len(input_len_list)):
                        expnum = str(p)
                        modeltype = ('AEDGAN_')
                        seq_len = input_len_list[i]
                        target_len = output_len_list[j]
                        gan = AEDGAN()
                        gan.train()  
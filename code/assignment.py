import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from preprocess import *


# #make custom RNN layer
# class RNNDecoderCell(tf.keras.layers.Layer):

#     def __init__(self, units):
#         self.units = units
#         self.state_size = units

#     def build(self, input_shape):
#         self.kernal = self.add_weight(shape=(input_shape[-1], self,.units), initializer='uniform', name='kernal')
#         self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform', name='recurrent_kernel')
#         self.built = True
    
#     def call(self, inputs, states):
#         prev_output = states[0]
#         h = K.dot(inputs, self.kernel)
#         output = h + K.dot(prev_output, self.recurrent_kernel)
#         return output, [output]



class Encoder_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        #define optimizer, layers, hyperparameters
        self.vocab_size = vocab_size
        self.embedding_size = 100
        self.dim_y = 200
        self.dim_z = 500
        self.dim_h = self.dim_y + self.dim_z
        self.n_layers = 1
        self.learning_rate = 0.0005

        #possibly only want to use one embedding layer --> in which case should probably merge encoder and generator classes?
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.dense1 = tf.keras.layers.Dense(self.dim_y)
        self.gru1 = tf.keras.layers.GRU(self.dim_h, dropout=0.5, return_sequences=True, return_state=True)

        self.dense2 = tf.keras.layers.Dense(self.dim_y)
        pass
    
    @tf.function
    def call(self, batch):
        labels = tf.reshape(batch['labels'], [-1, 1])

        enc_inputs = self.embedding_layer(batch['enc_inputs'])

        init_state = tf.concat([self.dense1(labels), tf.zeros([batch['size'], self.dim_z])], 1)
        _, z = self.gru1(enc_inputs, initial_state=init_state)
        z = z[:, self.dim_y:]


        return z
        pass
    
    @tf.function
    def loss_function(self):
        pass

class Generator_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        #define optimizer, layers, hyperparameters
        self.vocab_size = vocab_size
        self.embedding_size = 100
        self.dim_y = 200
        self.dim_z = 500
        self.dim_h = self.dim_y + self.dim_z
        self.n_layers = 1
        self.learning_rate = 0.0005
        self.dropout = 0.5


        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.dense1 = tf.keras.layers.Dense(self.dim_y)
        self.gru1 = tf.keras.layers.GRU(self.dim_h, dropout=self.dropout, return_sequences=True, return_state=True)

        self.projection = tf.keras.layers.Dense(self.vocab_size)
        pass
    
    @tf.function
    def call(self, batch, z):
        labels = tf.reshape(batch['labels'], [-1, 1])

        dec_inputs = self.embedding_layer(batch['dec_inputs'])

        self.h_ori = tf.concat([self.dense1(labels), z], 1)
        self.h_tsf = tf.concat([self.dense1(1-labels), z], 1)

        g_outputs, _ = self.gru1(dec_inputs, initial_state=self.h_ori)

        #attach h0 to the front

        teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        g_outputs = tf.reshape(g_outputs, [-1, self.dim_h])
        g_logits = self.projection(g_outputs)


        go = dec_inputs[:, 0,:]
        soft_func = softsample_word(self.dropout, )


        # return teach_h, g_logits
    
    @tf.function
    def loss_function(self, batch, g_logits):
        loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(batch['targets'], [-1]), logits=g_logits)
        loss_rec *= tf.reshape(batch['weights'], [-1])
        loss_rec = tf.reduce_sum(loss_rec) / tf.to_float(batch['size'])
        return loss_rec

class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        #define optimizer, layers, hyperparameters
        pass
    
    @tf.function
    def call(self, inputs):
        pass
    
    @tf.function
    def loss_function(self):
        pass

def train(batch, encoder, generator, discriminator_1, discriminator_2):
    with tf.GradientTape() as tape:
        z = encoder.call(batch)
        teach_h, g_logits = generator.call(batch, z)
        rec_loss = generator.loss_function(batch, g_logits)
        discriminator_1(teach_h, g_logits)


    pass

def main():

    #get training and testing data, vocab
    file_loc = "data/yelp/sentiment."
    train0, train1, test0, test1, vocab = get_data(file_loc + 'train.0', file_loc + 'train.1', file_loc + "test.0", file_loc + "test.1")
    batch_size = 64
    batches = get_batches(train0, train1, vocab, batch_size)
    # Initialize model
    encoder = Encoder_Model(vocab_size = len(vocab))
    generator = Generator_Model(vocab_size = len(vocab))
    discriminator_1 = Discriminator_Model()
    discriminator_2 = Discriminator_Model()

    max_epochs = 5
    #train etc
    
    for epoch in range(0, max_epochs):
        for batch in batches:
            train(batch, encoder, generator, discriminator_1, discriminator_2)

if __name__ == '__main__':
    main()
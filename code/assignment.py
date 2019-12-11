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

def gumbel_softmax(logits, gamma, eps=1e-20):
    U = tf.random.uniform(tf.shape(logits))
    G = -tf.math.log(-tf.math.log(U + eps) + eps)
    return tf.nn.softmax((logits + G) / gamma)

def softsample_word(dropout, dense, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = dense(output)
        prob = gumbel_softmax(logits, gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

class DecoderRNN(tf.keras.layers.Layer):
        def __init__(self, rnn_size, vocab_size, dropout):
            super(DecoderRNN, self).__init__()
            self.rnn_size = rnn_size
            self.vocab_size = vocab_size
            self.dropout = dropout
            self.dense = tf.keras.layers.Dense(self.vocab_size)

        @tf.function
        def call(self, h, inp, length, cell, embedding, gamma):
            gru_input = inp
            h_seq = []
            logits_seq = []

            h = tf.expand_dims(h, 0)
            for t in range(length):
                h = tf.convert_to_tensor(h)

                h_seq.append(tf.expand_dims(h, 1))

                output, h = cell(gru_input, h)
                # gru_input, logits = loop_func(output)


                output = tf.nn.dropout(output, self.dropout)
                logits = self.dense(output)
                prob = gumbel_softmax(logits, gamma)
                gru_input = tf.matmul(prob, embedding)



                logits_seq.append(tf.expand_dims(logits, 1))

            return tf.concat(h_seq, 1), tf.concat(logits_seq, 1)

#combined both
class Encoder_Decoder_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Encoder_Decoder_Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

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
       


        self.dense2 = tf.keras.layers.Dense(self.dim_y)
        self.gru2 = tf.keras.layers.GRU(self.dim_h, dropout=self.dropout, return_sequences=True, return_state=True)

        self.projection = tf.keras.layers.Dense(self.vocab_size)
        
        self.cell = tf.keras.layers.GRUCell(units = self.dim_h, dropout=self.dropout)

        self.decoderRNN = DecoderRNN(self.dim_h, self.vocab_size, dropout=self.dropout)
        self.gru3 = tf.keras.layers.GRU(self.dim_h, dropout=self.dropout, return_sequences=True, return_state=True)


    
    @tf.function
    def call(self, batch):
        labels = tf.reshape(batch['labels'], [-1, 1])
        enc_inputs = tf.convert_to_tensor(batch['enc_inputs'])
        enc_inputs = self.embedding_layer(enc_inputs)

        init_state = tf.concat([self.dense1(labels), tf.zeros([batch['size'], self.dim_z])], 1)
        _, z = self.gru1(enc_inputs, initial_state=init_state)
        z = z[:, self.dim_y:]



        #generator/decoder
        dec_inputs = tf.convert_to_tensor(batch['dec_inputs'])
        dec_inputs = self.embedding_layer(dec_inputs)

        self.h_ori = tf.concat([self.dense2(labels), z], 1)
        self.h_tsf = tf.concat([self.dense2(1-labels), z], 1)

        g_outputs, _ = self.gru2(dec_inputs, initial_state=self.h_ori)

        #attach h0 to the front

        teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        g_outputs = tf.reshape(g_outputs, [-1, self.dim_h])
        g_logits = self.projection(g_outputs)

        # go = dec_inputs[:, 0,:]
        # soft_func = softsample_word(self.dropout, self.projection, self.embedding_layer.embeddings, 0.1)
        # soft_h_tsf, soft_logits_tsf = self.decoderRNN(self.h_tsf, go, 20, self.cell, self.embedding_layer.embeddings, 0.1)
        soft_h_tsf, second = self.gru3(dec_inputs, initial_state=self.h_tsf)

        return teach_h, g_logits, soft_h_tsf

    @tf.function
    def loss_function(self, batch, g_logits):
        loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(batch['targets'], [-1]), logits=g_logits)
        loss_rec *= tf.reshape(batch['weights'], [-1])
        loss_rec = tf.reduce_sum(loss_rec) / tf.cast(batch['size'], dtype=tf.float32)
        return loss_rec

#######Separate######

class Encoder_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Encoder_Model, self).__init__()
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
    
    @tf.function
    def call(self, batch):
        labels = tf.reshape(batch['labels'], [-1, 1])
        enc_inputs = tf.convert_to_tensor(batch['enc_inputs'])

        enc_inputs = self.embedding_layer(enc_inputs)

        init_state = tf.concat([self.dense1(labels), tf.zeros([batch['size'], self.dim_z])], 1)
        _, z = self.gru1(enc_inputs, initial_state=init_state)
        z = z[:, self.dim_y:]


        return z
    
    @tf.function
    def loss_function(self):
        pass

class Generator_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Generator_Model, self).__init__()
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
        self.decoderRNN = DecoderRNN(self.dim_h, self.dropout)
        pass
    
    @tf.function
    def call(self, batch, z):
        labels = tf.reshape(batch['labels'], [-1, 1])
        dec_inputs = tf.convert_to_tensor(batch['dec_inputs'])

        dec_inputs = self.embedding_layer(dec_inputs)

        self.h_ori = tf.concat([self.dense1(labels), z], 1)
        self.h_tsf = tf.concat([self.dense1(1-labels), z], 1)

        g_outputs, _ = self.gru1(dec_inputs, initial_state=self.h_ori)

        #attach h0 to the front
        print('HELLO??')
        print(g_outputs.shape)
        print(self.h_ori.shape)
        teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        g_outputs = tf.reshape(g_outputs, [-1, self.dim_h])
        g_logits = self.projection(g_outputs)

        print("DEC INPUT SHAPE")
        print(dec_inputs.shape)
        go = dec_inputs[:, 0,:]
        print(go.shape)

        soft_func = softsample_word(self.dropout, self.projection, self.embedding_layer.embeddings, 0.1)
        print("YOOOOO")
        print(self.h_tsf.shape)
        soft_h_tsf, soft_logits_tsf = self.decoderRNN(self.h_tsf, go, 20, soft_func)
        soft_h_tsf = soft_h_tsf[:, :1+batch['len'], :]
        return teach_h, g_logits, soft_h_tsf
        # return teach_h, g_logits
    
    @tf.function
    def loss_function(self, batch, g_logits):
        loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(batch['targets'], [-1]), logits=g_logits)
        loss_rec *= tf.reshape(batch['weights'], [-1])
        loss_rec = tf.reduce_sum(loss_rec) / tf.cast(batch['size'], dtype=tf.float32)
        return loss_rec

class Discriminator_Model(tf.keras.Model):
    def __init__(self, batch_size):
        super(Discriminator_Model, self).__init__()
        #define optimizer, layers, hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)


        self.n_filters = 128
        self.batch_size = batch_size
        self.cnn1 = tf.keras.layers.Conv2D(self.n_filters, [1, batch_size], 1, padding='VALID')
        self.maxpool1 = tf.keras.layers.MaxPool2D()
        self.cnn2 = tf.keras.layers.Conv2D(self.n_filters, [2, batch_size], 1, padding='VALID')
        self.maxpool2 = tf.keras.layers.MaxPool2D()

        self.norm2 = tf.keras.layers.BatchNormalization()

        self.cnn3 = tf.keras.layers.Conv2D(self.n_filters, [3, batch_size], 1, padding='VALID')
        self.maxpool3 = tf.keras.layers.MaxPool2D()

        self.norm3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(1)

        self.norm4 = tf.keras.layers.BatchNormalization()


        pass
    
    @tf.function
    def call(self, x_real, x_fake):

        x_real = tf.expand_dims(x_real, 3)
        
        d_real = self.cnn1(x_real)
        d_real = tf.nn.leaky_relu(d_real)
        d_real = self.maxpool1(d_real)


        # d_real = tf.reduce_max(d_real, axis=1)

        d_real = self.cnn2(d_real)
        d_real = self.norm2(d_real)
        d_real = tf.nn.leaky_relu(d_real)
        d_real = self.maxpool2(d_real)

        # d_real = tf.reduce_max(d_real, axis=1)

        d_real = self.cnn3(d_real)
        d_real = self.norm3(d_real)
        d_real = tf.nn.leaky_relu(d_real)
        
        d_real = tf.nn.dropout(d_real, 0.5)

        d_real = self.flatten(d_real)

        d_real = self.dense1(d_real)
        d_real = self.norm4(d_real)

        d_real = tf.reshape(d_real, [-1])



        x_fake = tf.expand_dims(x_fake, 3)

        d_fake = self.cnn1(x_fake)
        d_fake = tf.nn.leaky_relu(d_fake)
        d_fake = self.maxpool1(d_fake)

        # # d_fake = tf.reduce_max(d_fake, axis=1)

        d_fake = self.cnn2(d_fake)
        d_fake = self.norm2(d_fake)
        d_fake = tf.nn.leaky_relu(d_fake)
        d_fake = self.maxpool2(d_fake)
        # # d_fake = tf.reduce_max(d_fake, axis=1)


        d_fake = self.cnn3(d_fake)
        d_fake = self.norm3(d_fake)
        d_fake = tf.nn.leaky_relu(d_fake)
        # # d_fake = tf.reduce_max(d_fake, axis=1)


        d_fake = tf.nn.dropout(d_fake, 0.5)
        d_fake = self.flatten(d_fake)
        d_fake = self.dense2(d_fake)
        d_fake = self.norm4(d_fake)


        d_fake = tf.reshape(d_fake, [-1])

        return d_real, d_fake

    
    @tf.function
    def loss_function(self, d_real, d_fake, ones, zeros):
        ones = tf.cast(ones, tf.float32)
        zeros = tf.cast(zeros, tf.float32)
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ones, logits=d_real)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=zeros, logits=d_fake))
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ones, logits=d_fake))

        return loss_d, loss_g

def train(batch, enc_dec, discriminator_0, discriminator_1):
    half = batch['size'] // 2
    print(half)
    print(batch['len'])
    zeros, ones = batch['labels'][:half], batch['labels'][half:]

    with tf.GradientTape(persistent=True) as enc_dec_tape:
        teach_h, g_logits, soft_h_tsf = enc_dec.call(batch)
        rec_loss = enc_dec.loss_function(batch, g_logits)
        d_real_0, d_fake_0 = discriminator_0.call(teach_h[:half], soft_h_tsf[half:])
        d_real_1, d_fake_1 = discriminator_1.call(teach_h[half:], soft_h_tsf[:half])
        loss_d_0, loss_g_0 = discriminator_0.loss_function(d_real_0, d_fake_0, ones, zeros)
        loss_d_1, loss_g_1 = discriminator_1.loss_function(d_real_1, d_fake_1, ones, zeros)

        loss_adv = loss_g_0 + loss_g_1

        final_loss = rec_loss + loss_adv
        print("LOSS")
        print(final_loss)

    enc_dec_gradients = enc_dec_tape.gradient(final_loss, enc_dec.trainable_variables)
    enc_dec.optimizer.apply_gradients(zip(enc_dec_gradients, enc_dec.trainable_variables))
    # enc_dec.optimizer.minimize(rec_loss, enc_dec.trainable_variables)
    # eg_gradients = eg_tape.gradient(final_loss, [encoder.trainable_variables, generator.trainable_variables])
    # optimizer.apply_gradients(zip(eg_gradients, [encoder.trainable_variables, generator.trainable_variables]))
    # optimizer.minimize(rec_loss, [encoder.trainable_variables, generator.trainable_variables])

    d0_gradients = enc_dec_tape.gradient(loss_d_0, discriminator_0.trainable_variables)
    discriminator_0.optimizer.apply_gradients(zip(d0_gradients, discriminator_0.trainable_variables))

    # discriminator_0.optimizer.minimize(loss_d_0, discriminator_0.trainable_variables)

    d1_gradients = enc_dec_tape.gradient(loss_d_1, discriminator_1.trainable_variables)
    discriminator_1.optimizer.apply_gradients(zip(d1_gradients, discriminator_1.trainable_variables))

    # discriminator_1.optimizer.minimize(loss_d_1, discriminator_1.trainable_variables)

    pass

def main():

    #get training and testing data, vocab
    file_loc = "data/yelp/sentiment."
    train0, train1, test0, test1, vocab = get_data(file_loc + 'train.0', file_loc + 'train.1', file_loc + "test.0", file_loc + "test.1")
    batch_size = 64
    batches = get_batches(train0, train1, vocab, batch_size)
    # Initialize model

    # encoder = Encoder_Model(vocab_size = len(vocab))
    # generator = Generator_Model(vocab_size = len(vocab))

    enc_dec = Encoder_Decoder_Model(vocab_size=len(vocab))
    discriminator_1 = Discriminator_Model(batch_size)
    discriminator_2 = Discriminator_Model(batch_size)

    max_epochs = 5
    #train etc
    
    for epoch in range(0, max_epochs):
        for batch in batches:
            train(batch, enc_dec, discriminator_1, discriminator_2)

if __name__ == '__main__':
    main()
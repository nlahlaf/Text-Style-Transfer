import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf

from preprocess import *

import tensorflow_gan as tfgan
import tensorflow_hub as hub

import numpy as np

from imageio import imwrite
import os
import argparse

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

parser = argparse.ArgumentParser(description='aligned')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--out-dir', type=str, default='./output',
                    help='Data where sampled output images will be written')

parser.add_argument('--log-every', type=int, default=50,
                    help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=100,
                    help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()

class Encoder_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Encoder_Model, self).__init__()
        #define optimizer, layers, hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

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

        self.projection = tf.keras.layers.Dense(self.vocab_size)
        self.gru3 = tf.keras.layers.GRU(self.dim_h, dropout=self.dropout, return_sequences=True, return_state=True)
        pass
    
    @tf.function
    def call(self, batch, z, transfer=False):
        labels = tf.reshape(batch['labels'], [-1, 1])
        dec_inputs = tf.convert_to_tensor(batch['dec_inputs'])

        dec_inputs = self.embedding_layer(dec_inputs)

        if transfer:
            self.h_tsf = tf.concat([self.dense1(1-labels), z], 1)
            g_outputs, _ = self.gru1(dec_inputs, initial_state=self.h_tsf)
        else:
            self.h_ori = tf.concat([self.dense1(labels), z], 1)
            g_outputs, _ = self.gru1(dec_inputs, initial_state=self.h_ori)
        

        #attach h0 to the front
        #teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)

        g_outputs = tf.reshape(g_outputs, [-1, self.dim_h])

        g_logits = self.projection(g_outputs)

        return g_logits
    
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

        self.dense1_x1 = tf.keras.layers.Dense(100, activation="relu")
        self.dense2_x1 = tf.keras.layers.Dense(1)

        self.dense1_x2 = tf.keras.layers.Dense(100, activation="relu")
        self.dense2_x2 = tf.keras.layers.Dense(1)


        pass
    
    @tf.function
    def call(self, z1, z2):

        z1_out = self.dense1_x1(z1)
        z1_out = self.dense2_x1(z1_out)
        z1_out = tf.keras.activations.sigmoid(z1_out)

        z2_out = self.dense1_x2(z2)
        z2_out = self.dense2_x2(z2_out)
        z2_out = tf.keras.activations.sigmoid(z2_out)

        return z1_out, z2_out

    
    @tf.function
    def loss_function(self, d_z1, d_z2):

        loss = tf.reduce_mean(tf.losses.BinaryCrossentropy().call(d_z1, d_z2))

        return loss

def train(batch, encoder, generator, discriminator, iteration, manager):
    half = batch['size'] // 2

    with tf.GradientTape() as enc_dec_tape:
        z = encoder.call(batch)
        g_logits = generator.call(batch, z)
        rec_loss = generator.loss_function(batch, g_logits)

        half = int(g_logits.shape[0] / 2 )
        
        d_z1, d_z2 = discriminator.call(g_logits[:half], g_logits[half:])
        d_loss = discriminator.loss_function(d_z1, d_z2)

        final_loss = rec_loss - d_loss
        if iteration % 10 == 0:
            print("LOSS")
            print(final_loss)

    enc_grads = enc_dec_tape.gradient(final_loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(enc_grads, generator.trainable_variables))

    with tf.GradientTape() as enc_dec_tape:
        z = encoder.call(batch)
        g_logits = generator.call(batch, z)
        rec_loss = generator.loss_function(batch, g_logits)
        half = int(g_logits.shape[0] / 2 )
        
        d_z1, d_z2 = discriminator.call(g_logits[:half], g_logits[half:])
        d_loss = discriminator.loss_function(d_z1, d_z2)
        final_loss = rec_loss - d_loss

    d_grads = enc_dec_tape.gradient(final_loss, discriminator.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    if iteration % args.save_every == 0:
        manager.save()

    pass

def transfer(encoder, generator, id2word, batch):
    z = encoder.call(batch)
    g_logits = generator.call(batch, z, transfer=True)
    g_logits = tf.reshape(g_logits, [batch["size"], batch["len"], -1])

    half = int(g_logits.shape[0] / 2 )
    ori_logits = g_logits[:half]
    tsf_logits = g_logits[half:]
    print("Original style 1 sentences")
    print([[id2word[i] for i in sent] for sent in batch["dec_inputs"][half:(half+10)]])
    tsf = np.argmax(tsf_logits, axis=2).tolist()
    tsf = [[id2word[i] for i in sent] for sent in tsf[0:10]]
    print("Style 1 sentences transferred to Style 2")
    print(tsf)

    print("Original style 2 sentences")
    print([[id2word[i] for i in sent] for sent in batch["dec_inputs"][0:10]])
    ori = np.argmax(ori_logits, axis=2).tolist()
    ori = [[id2word[i] for i in sent] for sent in ori[0:10]]
    print("Style 2 sentences transferred to style 1")
    print(ori)

    return tsf

def main():

    #get training and testing data, vocab
    file_loc = "data/yelp/sentiment."
    train0, train1, test0, test1, vocab, id2word = get_data(file_loc + 'train.0', file_loc + 'train.1', file_loc + "test.0", file_loc + "test.1")
    batch_size = 64
    batches = get_batches(train0, train1, vocab, batch_size)
    # Initialize model

    encoder = Encoder_Model(vocab_size = len(vocab))
    generator = Generator_Model(vocab_size = len(vocab))

    discriminator = Discriminator_Model(batch_size)

    max_epochs = 10
    #train etc
    
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.restore_checkpoint:
        # restores the latest checkpoint using from the manager
        print("RESTORING CHECKPOINT")
        checkpoint.restore(manager.latest_checkpoint) 

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            for epoch in range(0, max_epochs):
                print('========================== EPOCH %d  ==========================' % epoch)
                for iteration, batch in enumerate(batches):
                    train(batch, encoder, generator, discriminator, iteration, manager)
                    # if iteration % args.save_every == 0:
                    if iteration % 10 == 0:
                        transfer(encoder, generator, id2word, batch)
                        # print(tsf[0:10])
                # print("Average FID for Epoch: " + str(avg_fid))
                # Save at the end of the epoch, too
                print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                manager.save()
    except RuntimeError as e:
        print(e)

    #done training
    print("DONE TRAINING")

    test_batches = get_batches(test0, test1, vocab, batch_size)
    for iteration, batch in enumerate(test_batches):
        tsf = transfer(encoder, generator, id2word, batch)
        print(tsf[0:10])



if __name__ == '__main__':
    main()

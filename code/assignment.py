import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from preprocess import *


def main():

    #get training and testing data, vocab
    file_loc = "data/yelp/sentiment."
    train0, train1, test0, test1, vocab = get_data(file_loc + 'train.0', file_loc + 'train.1', file_loc + "test.0", file_loc + "test.1")

    # Initialize model
    #train etc

if __name__ == '__main__':
    main()

`
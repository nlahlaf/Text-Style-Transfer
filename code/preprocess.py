import numpy as np
import tensorflow as tf
from collections import Counter


########## #####################
PAD_TOKEN = '<pad>'
GO_TOKEN = '<go>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
########## #####################

def build_vocab(sentences, min_occur=5):
    """
    Builds vocab from list of sentences

    :param sentences:  list of sentences, each a list of words
    :param min_occur: minimum number of total occurances required for a word to qualify for inclusion in the vocab
    :return: tuple of (vocab dictionary: word --> unique index, list of all words, vocab_size)
    """
    #create word2id-> map pf each word/token to its number ID in the vocab
    word2id = {PAD_TOKEN:0, GO_TOKEN:1, EOS_TOKEN:2, UNK_TOKEN:3}

    #create id2word-> list of all tokens/words indexed by their ID
    #id2word = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, UNK_TOKEN]

    words = [word for sent in sentences for word in sent] #list of all words in all sentences
    cnt = Counter(words) #counts occurances
    for word in cnt:
        if cnt[word] >= min_occur:
            word2id[word] = len(word2id)
    #        id2word.append(word)
    vocab_size = len(word2id)

    return word2id


def read_data(file_name):
	"""
  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt') as data_file:
		for line in data_file: text.append(line.split())
	return text

def get_batch(x, y, word2id, min_len=5):
    pad = word2id[PAD_TOKEN]
    go = word2id[GO_TOKEN]
    eos = word2id[EOS_TOKEN]
    unk = word2id[UNK_TOKEN]

    enc_inputs, dec_inputs, targets, weights = [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        enc_inputs.append(padding + sent_id[::-1])
        dec_inputs.append([go] + sent_id + padding)
        targets.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))

    return {'enc_inputs': enc_inputs,
            'dec_inputs': dec_inputs,
            'targets':    targets,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1}

def makeup(_x, n):
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x

def get_batches(x0, x1, word2id, batch_size):
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
    n = len(x0)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(x0[s:t] + x1[s:t],
            [0]*(t-s) + [1]*(t-s), word2id))
        s = t

    return batches

def get_data(train0_file, train1_file, test0_file, test1_file):
	"""
	inputs: files for training sentiments 0 and 1, testing sentiments 0 and 1

	returns: list of training sentences (0 and 1 sentiments)
	.........list of testing sentences (0 and 1 sentiments)
	.........the vocab
	"""
	train0 = read_data(train0_file)
	train1 = read_data(train1_file)

	test0 = read_data(test0_file)
	test1 = read_data(test1_file)

	vocab = build_vocab(train0 + train1)

	return train0, train1, test0, test1, vocab

	

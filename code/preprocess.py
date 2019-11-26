import numpy as np
import tensorflow as tf

def read_data(file_name):
	"""
  Load text data from file; separate into sentences and then words

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt') as data_file:
		for line in data_file: text.append(line.split())
	return text

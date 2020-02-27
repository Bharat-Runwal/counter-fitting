import sys
import os
import itertools
import operator

import numpy as np
import torch
from absl import flags
from absl import app

FLAGS = flags.FLAGS

def add_flags():
	flags.DEFINE_string('pretrained_vectors', 'word_vectors/glove.txt', 'Location of pretrained word embeddings', short_name = 'vec')
	
	flags.DEFINE_spaceseplist('antonyms', ['linguistic_constraints/ppdb_antonyms.txt', 'linguistic_constraints/wordnet_antonyms.txt'], 'List of files storing antonym pairs.')
	
	flags.DEFINE_spaceseplist('synonyms', ['linguistic_constraints/ppdb_synonyms.txt'], 'List of files storing synonym pairs.')

	flags.DEFINE_spaceseplist('ontology_filepath', [], 'List of dialogue ontology files to be used to extract additional antonyms.')
	
	flags.DEFINE_string('vectors_out', 'counterfitted_vectors.txt', 'Where to store counterfitted vectors.', short_name = 'o')

	flags.DEFINE_bool('train', True, 'Whether to run the counterfitting experiment.')

	flags.DEFINE_integer('train_steps', 10000, 'How many sampling steps to run.')

	flags.DEFINE_bool('evaluate', True, 'Whether to run evaluation on SimLex-999.')

	flags.DEFINE_string('simlex', 'linguistic_constraints/SimLex-999.txt', 'Location of the SimLex-999 file.')

	flags.DEFINE_float('k1', .1, 'Weight of antonym loss.')
	
	flags.DEFINE_float('k2', .1, 'Weight of synonym loss.')
	
	flags.DEFINE_float('k3', .1, 'Weight of negative sample pair loss.')
	
	flags.DEFINE_float('learning_rate', 1, 'Learning rate.', short_name = 'lr')


def load_vectors(filename, normalize = True):
	if not os.path.exists(filename):
		raise ValueError('Word vectors file {} not found.'.format(filename))
	vector_dict = dict()
	
	def _proc_line(line):
		nonlocal vector_dict
		line = line.strip().split()
		vector_dict[line[0]] = np.asarray([float(x) for x in line[1:]])
		if normalize:
			vector_dict[line[0]] = vector_dict[line[0]] / np.linalg.norm(vector_dict[line[0]])
	
	print("Loading pretrained word vectors from", filename)
	with open(filename) as fin:
		firstline = fin.readline()
		if len(firstline.strip().split()) > 2:
			_proc_line(firstline)
		for line in fin:
			_proc_line(line)

	return vector_dict


def print_word_vectors(word_vectors, write_path):
	"""
	This function prints the collection of word vectors to file, in a plain textual format. 
	"""
	print("Saving the counter-fitted word vectors to", write_path, "\n")
	with open(write_path, "w") as f_write:
		for key in word_vectors:
			f_write.write(key + ' ' + ' '.join(map(str, numpy.round(word_vectors[key], decimals=6))) + '\n')


def load_constraints(constraints_filepaths, vocabulary):
	"""
	This methods reads a collection of constraints from the specified file, and returns a set with
	all constraints for which both of their constituent words are in the specified vocabulary.
	"""
	constraints = []
	for constraints_file in constraints_filepaths:
		with open(constraints_file) as f:
			for line in f:
				word_pair = sorted(line.split())
				if word_pair[0] in vocabulary and word_pair[1] in vocabulary and word_pair[0] != word_pair[1]:
					constraints.append(tuple(word_pair))

	constraints = set(constraints)
	print(constraints_filepaths, "yielded", len(constraints), "constraints.")

	return constraints


def extract_antonyms_from_dialogue_ontology(dialogue_ontology, vocabulary):
	"""
	Returns a list of antonyms for the supplied dialogue ontology, which needs to be provided as a dictionary.
	The dialogue ontology must follow the DST Challenges format: we only care about goal slots, i.e. informables.
	"""
	# We are only interested in the goal slots of the ontology:
	dialogue_ontology = dialogue_ontology["informable"]

	slot_names = set(dialogue_ontology.keys())

	# Forcing antonymous relations between different entity names does not make much sense. 
	if "name" in slot_names:
		slot_names.remove("name")

	# Binary slots - we do not know how to handle - there is no point enforcing antonymy relations there. 
	binary_slots = set()
	for slot_name in slot_names:
		current_values = dialogue_ontology[slot_name]
		if len(current_values) == 2 and "true" in current_values and "false" in current_values:
			binary_slots |= {slot_name}

	if binary_slots:
		print("Removing binary slots:", binary_slots)
	else:
		print("There are no binary slots to ignore.")

	slot_names = slot_names - binary_slots

	antonym_list = set()

	# add antonymy relations between each pair of slot values for each non-binary slot. 
	for slot_name in slot_names:
		current_values = dialogue_ontology[slot_name]
		for index_1, value in enumerate(current_values):
			for index_2 in range(index_1 + 1, len(current_values)):
				# note that this will ignore all multi-value words. 
				if value in vocabulary and current_values[index_2] in vocabulary:
					antonym_list |= {tuple(sorted(value, current_values[index_2]))}

	return antonym_list


def preproc_vectors(vector_dict, leaveout_vocab,  max_mem = 4e9):
	wordembs = torch.tensor(list(vector_dict.values()), dtype = torch.float32)
	word2idx = {w:i for i, w in enumerate(vector_dict.keys())}
	idx2word = {i:w for w,i in word2idx.items()}
	
	validembs = {w:vector_dict[w] for w in vector_dict if w not in leaveout_vocab}
	val_idx2word = {i:w for i, w in enumerate(validembs.keys())}
	validembs = torch.tensor(list(validembs.values()), dtype = torch.float32)
	
	batchsize = int(max_mem / (len(validembs) * 4))
	print('batchsize is', batchsize)
	nearest_neighbor = dict()
	counter = 0
	for i_start in range(0, len(wordembs), batchsize):
		counter+=1
		sys.stdout.write('\rProcessing batch {}.'.format(counter))

		chunksize = min(len(wordembs) - i_start, batchsize)
		nns = torch.matmul(wordembs[i_start:i_start+chunksize, :], torch.t(validembs))
		nns = nns - (nns > 1-1e-3).float()
		nns = torch.argmax(nns, dim = 0)
		for j in range(chunksize):
			nearest_neighbor[idx2word[i_start+j]] = val_idx2word[int(nns[j])]
	print()
	return wordembs, word2idx, idx2word, nearest_neighbor



def main(_):
	vdict = load_vectors(FLAGS.pretrained_vectors)
	vocab = set(vdict.keys())
	antonyms = load_constraints(FLAGS.antonyms, vocab)
	synonyms = load_constraints(FLAGS.synonyms, vocab)
	leaveout_vocab = set([f(x) for x in antonyms | synonyms for f in [operator.itemgetter(0), operator.itemgetter(1)]])
	wordembs, word2idx, idx2word, nearest_neighbor = preproc_vectors(vdict, leaveout_vocab)
	print(nearest_neighbor)
	


if __name__ == '__main__':
	add_flags()
	app.run(main)

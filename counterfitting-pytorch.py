import sys
import os
import itertools
import operator
import random
import json

import numpy as np
import torch
from absl import flags
from absl import app
from scipy import stats

FLAGS = flags.FLAGS

def add_flags():
	flags.DEFINE_string('pretrained_vectors', 'word_vectors/glove.txt', 'Location of pretrained word embeddings', short_name = 'vec')
	
	flags.DEFINE_spaceseplist('antonyms', ['linguistic_constraints/ppdb_antonyms.txt', 'linguistic_constraints/wordnet_antonyms.txt'], 'List of files storing antonym pairs.')
	
	flags.DEFINE_spaceseplist('synonyms', ['linguistic_constraints/ppdb_synonyms.txt'], 'List of files storing synonym pairs.')

	flags.DEFINE_spaceseplist('ontology_files', [], 'List of dialogue ontology files to be used to extract additional antonyms.')
	
	flags.DEFINE_string('vectors_out', 'counterfitted_vectors.txt', 'Where to store counterfitted vectors.', short_name = 'o')

	flags.DEFINE_bool('train', True, 'Whether to run the counterfitting experiment.')

	flags.DEFINE_integer('train_steps', 100000, 'How many sampling steps to run.')

	flags.DEFINE_bool('evaluate', True, 'Whether to run evaluation on SimLex-999.')

	flags.DEFINE_string('simlex', 'linguistic_constraints/SimLex-999.txt', 'Location of the SimLex-999 file.')

	flags.DEFINE_float('delta_sim', 1, 'Ideal value of synonym cosine similarity.')
	
	flags.DEFINE_float('delta_ant', 0, 'Ideal value of antonym cosine similarity.')
	
	flags.DEFINE_float('lambda_reg', 1, 'Weight of L2 regularization.')
	
	flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.', short_name = 'lr')

	flags.DEFINE_bool('cuda', False, 'Whether to use cuda.')


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
			f_write.write(key + ' ' + ' '.join(map(str, word_vectors[key])) + '\n')


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


def preproc_vectors(vector_dict, leaveout_vocab,  max_mem = 6e9, device = torch.device('cpu')):
	wordembs = torch.tensor(list(vector_dict.values()), dtype = torch.float32, device = device)
	word2idx = {w:i for i, w in enumerate(vector_dict.keys())}
	idx2word = {i:w for w,i in word2idx.items()}
	
	validembs = {w:vector_dict[w] for w in vector_dict if w not in leaveout_vocab}
	val_idx2word = {i:w for i, w in enumerate(validembs.keys())}
	validembs = torch.tensor(list(validembs.values()), dtype = torch.float32, device = device)
	
	batchsize = int(max_mem / (len(validembs) * 4))
	print('Batchsize is', batchsize)
	nearest_neighbor = dict()
	farthest_vec = dict()
	counter = 0
	for i_start in range(0, len(wordembs), batchsize):
		counter+=1
		sys.stdout.write('\rProcessing batch {}.'.format(counter))

		chunksize = min(len(wordembs) - i_start, batchsize)
		nns = torch.matmul(wordembs[i_start:i_start+chunksize, :], torch.t(validembs))
		fvs = torch.argmin(nns, dim = 1)
		nns = nns - (nns > 1-1e-3).float()
		nns = torch.argmax(nns, dim = 1)
		for j in range(chunksize):
			nearest_neighbor[idx2word[i_start+j]] = val_idx2word[int(nns[j])]
			farthest_vec[idx2word[i_start+j]] = val_idx2word[int(fvs[j])]
	print()
	return wordembs, word2idx, idx2word, nearest_neighbor, farthest_vec


def train(wordembs, word2idx, nearest_neighbor, farthest_vec, antonyms, synonyms, flags_obj, device):
	addembs = torch.zeros(wordembs.size(), dtype = torch.float32, device = device, requires_grad = True)
	random.seed()
	antlist = list(antonyms)
	synlist = list(synonyms)

	def _activation(wordembs, other, delta, sample, sign = 1):
		nonlocal word2idx
		xl = wordembs[word2idx[sample[0]]]
		xr = wordembs[word2idx[sample[1]]]
		tl = wordembs[word2idx[other[sample[0]]]]
		tr = wordembs[word2idx[other[sample[1]]]]
		cos = lambda x,y: torch.nn.functional.cosine_similarity(x,y, dim = 0)
		act = torch.nn.functional.relu(delta + sign*(cos(xl, tl) - cos(xl, xr))) + torch.nn.functional.relu(delta + sign*(cos(xr, tr) - cos(xl, xr)))
		return act
	
	optimizer = torch.optim.Adam([addembs], lr = flags_obj.learning_rate, weight_decay = flags_obj.lambda_reg)
	da = flags_obj.delta_ant
	ds = flags_obj.delta_sim
	report_int = 100
	print('Training starting')
	pastlosses = []
	for step in range(flags_obj.train_steps):
		optimizer.zero_grad()
		antpair = random.choice(antlist)
		synpair = random.choice(synlist)
		loss = _activation(wordembs+addembs, farthest_vec, da, antpair, -1) + _activation(wordembs+addembs, nearest_neighbor, ds, synpair, 1)
		loss.backward()
		optimizer.step()
		pastlosses.append(float(loss))
		if (step + 1) % report_int == 0:
			print('Step {}: moving average loss {}.'.format(step + 1, np.average(pastlosses)))
			pastlosses = []

	addembs.requires_grad_(False)
	wordembs = wordembs + addembs
	return wordembs

def train_alt(wordembs, word2idx, sampling_vocab, antonyms, synonyms, flags_obj, device):
	addembs = torch.zeros(wordembs.size(), dtype = torch.float32, device = device, requires_grad = True)
	random.seed()
	antlist = list(antonyms)
	synlist = list(synonyms)
	sampling_vocab = list(sampling_vocab)

	def _activation(wordembs, delta, sample, sign = 1):
		nonlocal word2idx
		u = wordembs[word2idx[sample[0]]]
		w = wordembs[word2idx[sample[1]]]
		cos = lambda x,y: torch.nn.functional.cosine_similarity(x,y, dim = 0)
		act = torch.nn.functional.relu(sign*(cos(u, w) - delta))
		return act

	def _regularization(wordembs, addembs, sample, pivot):
		nonlocal word2idx
		uid = word2idx[sample[0]]
		wid = word2idx[sample[1]]
		pid = word2idx[pivot]
		cos = lambda x,y: torch.nn.functional.cosine_similarity(x,y, dim = 0)
		reg = (cos(wordembs[uid], wordembs[pid]) - cos((wordembs+addembs)[uid], wordembs[pid]))**2 + (cos(wordembs[wid], wordembs[pid]) - cos((wordembs+addembs)[wid], wordembs[pid]))**2
		return reg
	
	optimizer = torch.optim.Adam([addembs], lr = flags_obj.learning_rate, amsgrad = True)
	lrsched = torch.optim.lr_scheduler.StepLR(optimizer, 40000, gamma = .9)
	da = flags_obj.delta_ant
	ds = flags_obj.delta_sim
	lreg = flags_obj.lambda_reg
	report_int = 1000
	eval_int = 20000
	print('Training starting')
	pastlosses = []
	for step in range(flags_obj.train_steps):
		optimizer.zero_grad()
		antpair = random.choice(antlist)
		synpair = random.choice(synlist)
		loss = _activation(wordembs+addembs, da, antpair, 1) + _activation(wordembs+addembs, ds, synpair, -1) + lreg*(_regularization(wordembs, addembs, antpair, random.choice(sampling_vocab)) + _regularization(wordembs, addembs, synpair, random.choice(sampling_vocab)))
		loss.backward()
		optimizer.step()
		pastlosses.append(float(loss))
		lrsched.step()
		if (step + 1) % report_int == 0:
			print('Step {}: moving average loss {}.'.format(step + 1, np.average(pastlosses)))
			pastlosses = []

		if (step + 1) % eval_int == 0:
			evalembs = (wordembs+addembs).detach().cpu().numpy()
			edict = {w:evalembs[word2idx[w]] for w in word2idx}
			print("Evaluation rho with SimLex:", simlex_analysis(edict, flags_obj.simlex))
			
	addembs.requires_grad_(False)
	wordembs = wordembs + addembs
	return wordembs


def simlex_analysis(word_vectors, simlexfile):
	"""
	This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors. 
	The method also prints the gold standard SimLex-999 ranking to results/simlex_ranking.txt, 
	and the ranking produced using the counter-fitted vectors to results/counter_ranking.txt 
	"""
	fread_simlex = open(simlexfile, "r")
	pair_list = []

	line_number = 0
	for line in fread_simlex:
		if line_number > 0:
			tokens = line.split()
			word_i = tokens[0]
			word_j = tokens[1]
			score = float(tokens[3])
			if word_i in word_vectors and word_j in word_vectors:
				pair_list.append( ((word_i, word_j), score) )
		line_number += 1

	pair_list.sort(key=lambda x: - x[1])

	f_out_simlex = open("results/simlex_ranking.txt", "w")
	f_out_counterfitting = open("results/counter_ranking.txt", "w")

	extracted_list = []
	extracted_scores = {}

	for (x,y) in pair_list:

		(word_i, word_j) = x
		current_distance = 1-np.dot(word_vectors[word_i], word_vectors[word_j]) 
		extracted_scores[(word_i, word_j)] = current_distance
		extracted_list.append(((word_i, word_j), current_distance))

	extracted_list.sort(key=lambda x: x[1])

	# print both the gold standard ranking and the produced ranking to files in the results folder:
	def parse_pair(pair_of_words):
		return str(pair_of_words[0] + ", " + str(pair_of_words[1]))

	for idx, element in enumerate(pair_list):
		clean_elem = str(parse_pair(element[0])) + " : " +  str(round(element[1], 2))
		f_out_simlex.write(str(idx) + " :" + clean_elem + '\n')

	for idx, element in enumerate(extracted_list):
		clean_elem = str(parse_pair(element[0])) + " : " + str(round(element[1], 2))
		f_out_counterfitting.write(str(idx) + " :" + clean_elem + '\n')

	spearman_original_list = []
	spearman_target_list = []

	for position_1, (word_pair, score_1) in enumerate(pair_list):
		score_2 = extracted_scores[word_pair]
		position_2 = extracted_list.index((word_pair, score_2))
		spearman_original_list.append(position_1)
		spearman_target_list.append(position_2)

	spearman_rho = stats.spearmanr(spearman_original_list, spearman_target_list)
	return round(spearman_rho[0], 3)

def main(_):
	
	if FLAGS.train:
		embsdict = load_vectors(FLAGS.pretrained_vectors)
		vocab = set(embsdict.keys())
		antonyms = load_constraints(FLAGS.antonyms, vocab)
		synonyms = load_constraints(FLAGS.synonyms, vocab)
		
		if FLAGS.ontology_files != []:
			for jf in FLAGS.ontology_files:
				with open(jf, 'rb') as jin:
					ont = json.load(jin)
					antonyms |= extract_antonyms_from_dialogue_ontology(ont, vocab)
		
		synonyms = synonyms - antonyms # to deal with some noise in the dataset
		if len(antonyms & synonyms) > 0:
			raise ValueError('Antonyms and synonyms cannot overlap')

		leaveout_vocab = set([f(x) for x in antonyms | synonyms for f in [operator.itemgetter(0), operator.itemgetter(1)]])
		
		# if FLAGS.cuda:
		# 	device = torch.device('cuda')
		# else:
		# 	device = torch.device('cpu')
		
		# wordembs, word2idx, idx2word, nearest_neighbor, farthest_vec = preproc_vectors(embsdict, leaveout_vocab, device = torch.device('cpu'))
		# wordembs = wordembs.to(device = 'cuda')
		# wordembs = train(wordembs, word2idx, nearest_neighbor, farthest_vec, antonyms, synonyms, FLAGS, device = torch.device('cuda'))


		wordembs = torch.tensor(list(embsdict.values()), dtype = torch.float32, device = torch.device('cuda'))
		word2idx = {w:i for i, w in enumerate(embsdict.keys())}

		wordembs = train_alt(wordembs, word2idx, vocab - leaveout_vocab, antonyms, synonyms, FLAGS, device = torch.device('cuda'))


		wordembs = wordembs.cpu().numpy()
		
		newembs = {w:wordembs[word2idx[w]] for w in word2idx}
		print_word_vectors(newembs, FLAGS.vectors_out)
	
	if FLAGS.evaluate:
		if not FLAGS.train:
			embsdict = load_vectors(FLAGS.vectors_out)
			word2idx = {w:i for i,w in enumerate(embsdict.keys())}
		else:
			embsdict = dict()
			for w in newembs:
				embsdict[w] = newembs[w] / np.linalg.norm(newembs[w])

		print("\nSimLex score (Spearman's rho coefficient) the counter-fitted vectors is:", simlex_analysis(embsdict, FLAGS.simlex), "\n")



if __name__ == '__main__':
	add_flags()
	app.run(main)

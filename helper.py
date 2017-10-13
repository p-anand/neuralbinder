from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
from six.moves import cPickle
from deepomics import neuralnetwork as nn
from deepomics import utils, saliency


def import_model(model):
	""" import models by model name """

	if model == 'affinity_conv_net':
		from model_zoo import affinity_conv_net as genome_model

	elif model == 'affinity_all_conv_net':
		from model_zoo import affinity_all_conv_net as genome_model

	elif model == 'affinity_residual_net':
		from model_zoo import affinity_residual_net as genome_model

	elif model == 'clip_conv_net':
		from model_zoo import clip_conv_net as genome_model

	elif model == 'clip_all_conv_net':
		from model_zoo import clip_all_conv_net as genome_model

	elif model == 'clip_residual_net':
		from model_zoo import clip_residual_net as genome_model

	return genome_model


def load_dataset_hdf5(file_path, dataset_name=None, ss_type='seq', rbp_index=None):

	def prepare_data(train, ss_type=None):

		seq = train['inputs'][:,:,:,:4]
		
		if ss_type == 'pu':     
			structure = train['inputs'][:,:,:,4:9]
			paired = np.expand_dims(structure[:,:,:,0], axis=3)
			unpaired = np.expand_dims(np.sum(structure[:,:,:,1:], axis=3), axis=3)
			seq = np.concatenate([seq, paired, unpaired], axis=3)
		
		elif ss_type == 'struct':    
			structure = train['inputs'][:,:,:,4:9]
			paired = np.expand_dims(structure[:,:,:,0], axis=3)
			HIME = structure[:,:,:,1:]
			seq = np.concatenate([seq, paired, HIME], axis=3)
		
		train['inputs']  = seq
		return train

	# open dataset
	experiments = None
	dataset = h5py.File(file_path, 'r')
	if not dataset_name:
		# load set A data
		X_train = np.array(dataset['X_train']).astype(np.float32)
		Y_train = np.array(dataset['Y_train']).astype(np.float32)
		X_valid = np.array(dataset['X_valid']).astype(np.float32)
		Y_valid = np.array(dataset['Y_valid']).astype(np.float32)    
		X_test = np.array(dataset['X_test']).astype(np.float32)
		Y_test = np.array(dataset['Y_test']).astype(np.float32)

		# expand dims of targets
		if rbp_index is not None:
			Y_train = Y_train[:,rbp_index]
			Y_valid = Y_valid[:,rbp_index]
			Y_test = Y_test[:,rbp_index]
	else:
		X_train = np.array(dataset['/'+dataset_name+'/X_train']).astype(np.float32)        
		Y_train = np.array(dataset['/'+dataset_name+'/Y_train']).astype(np.float32)
		X_valid = np.array(dataset['/'+dataset_name+'/X_valid']).astype(np.float32)       
		Y_valid = np.array(dataset['/'+dataset_name+'/Y_valid']).astype(np.float32)
		X_test = np.array(dataset['/'+dataset_name+'/X_test']).astype(np.float32)
		Y_test = np.array(dataset['/'+dataset_name+'/Y_test']).astype(np.float32)
	
	# expand dims of targets
	if len(Y_train.shape) == 1:
		Y_train = np.expand_dims(Y_train, axis=1)
		Y_valid = np.expand_dims(Y_valid, axis=1)
		Y_test = np.expand_dims(Y_test, axis=1)

	# add another dimension to make a 4d tensor
	X_train = np.expand_dims(X_train, axis=3).transpose([0, 2, 3, 1])
	X_test = np.expand_dims(X_test, axis=3).transpose([0, 2, 3, 1])
	X_valid = np.expand_dims(X_valid, axis=3).transpose([0, 2, 3, 1])

	# dictionary for each dataset
	train = {'inputs': X_train, 'targets': Y_train}
	valid = {'inputs': X_valid, 'targets': Y_valid}
	test = {'inputs': X_test, 'targets': Y_test}

	# parse secondary structure profiles
	train = prepare_data(train, ss_type)
	valid = prepare_data(valid, ss_type)
	test = prepare_data(test, ss_type)
	
	return train, valid, test
	


def process_data(train, valid, test, method='log_norm'):
	"""get the results for a single experiment specified by rbp_index.
	Then, preprocess the binding affinity intensities according to method.
	method: 
		clip_norm - clip datapoints larger than 4 standard deviations from the mean
		log_norm - log transcormation
		both - perform clip and log normalization as separate targets (expands dimensions of targets)
	"""

	def normalize_data(data, method):
		if method == 'standard':
			MIN = np.min(data)
			data = np.log(data-MIN+1)
			sigma = np.mean(data)
			data_norm = (data)/sigma      
			params = sigma
		if method == 'clip_norm':
			# standard-normal transformation
			significance = 4     
			std = np.std(data)
			index = np.where(data > std*significance)[0]
			data[index] = std*significance
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma      
			params = [mu, sigma]

		elif method == 'log_norm':
			# log-standard-normal transformation
			MIN = np.min(data)
			data = np.log(data-MIN+1)
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma      
			params = [MIN, mu, sigma]
		
		elif method == 'both':
			data_norm1, params = normalize_data(data, 'clip_norm')
			data_norm2, params = normalize_data(data, 'log_norm')
			data_norm = np.hstack([data_norm1, data_norm2])
		return data_norm, params
	

	# get binding affinities for a given rbp experiment
	Y_train = train['targets']
	Y_valid = valid['targets']
	Y_test = test['targets']
	
	# filter NaN 
	train_index = np.where(np.isnan(Y_train) == False)[0]
	valid_index = np.where(np.isnan(Y_valid) == False)[0]
	test_index = np.where(np.isnan(Y_test) == False)[0]
	Y_train = Y_train[train_index]
	Y_valid = Y_valid[valid_index]
	Y_test = Y_test[test_index]
	X_train = train['inputs'][train_index]
	X_valid = valid['inputs'][valid_index]
	X_test = test['inputs'][test_index]

	# normalize intenensities
	if method:
		Y_train, params_train = normalize_data(Y_train, method)                 
		Y_valid, params_valid = normalize_data(Y_valid, method)                 
		Y_test, params_test = normalize_data(Y_test, method)                 

	# store sequences and intensities
	train = {'inputs': X_train, 'targets': Y_train}
	valid = {'inputs': X_valid, 'targets': Y_valid}
	test = {'inputs': X_test, 'targets': Y_test}

	return train, valid, test
					 

def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print("making directory: " + path)

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print("making directory: " + outdir)
	return outdir


def split_rnacompete_dataset(data, targets, experiment, valid_frac):
	index = np.where(experiment == 'A')[0]
	num_seq = len(index)
	num_valid = int(num_seq*valid_frac)

	shuffle = np.random.permutation(num_seq)

	X_train = data[shuffle[num_valid:]]
	Y_train = targets[[shuffle[num_valid:]]]
	train = (X_train, Y_train)
	
	X_valid = data[shuffle[:num_valid]]
	Y_valid = targets[[shuffle[:num_valid]]]
	valid = (X_valid, Y_valid)
	
	test_index = np.where(experiment == 'B')[0]
	X_test = data[test_index]
	Y_test = targets[test_index]
	test = (X_test, Y_test)
	
	return train, valid, test 



def split_dataset(one_hot, labels, valid_frac=0.1, test_frac=0.2):
	"""split dataset into training, cross-validation, and test set"""
	
	def split_index(num_data, valid_frac, test_frac):
		# split training, cross-validation, and test sets

		train_frac = 1 - valid_frac - test_frac
		cum_index = np.array(np.cumsum([0, train_frac, valid_frac, test_frac])*num_data).astype(int)
		shuffle = np.random.permutation(num_data)
		train_index = shuffle[cum_index[0]:cum_index[1]]        
		valid_index = shuffle[cum_index[1]:cum_index[2]]
		test_index = shuffle[cum_index[2]:cum_index[3]]

		return train_index, valid_index, test_index

	
	# split training, cross-validation, and test sets
	num_data = len(one_hot)
	train_index, valid_index, test_index = split_index(num_data, valid_frac, test_frac)
	
	# split dataset
	train = (one_hot[train_index], labels[train_index,:])
	valid = (one_hot[valid_index], labels[valid_index,:])
	test = (one_hot[test_index], labels[test_index,:])
	indices = [train_index, valid_index, test_index]

	return train, valid, test, indices


def save_dataset_hdf5(save_path, train, valid, test):
	# save datasets as hdf5 file
	with h5py.File(save_path, "w") as f:
		dset = f.create_dataset("X_train", data=train[0].astype(np.float32), compression="gzip")
		dset = f.create_dataset("Y_train", data=train[1].astype(np.float32), compression="gzip")
		dset = f.create_dataset("X_valid", data=valid[0].astype(np.float32), compression="gzip")
		dset = f.create_dataset("Y_valid", data=valid[1].astype(np.float32), compression="gzip")
		dset = f.create_dataset("X_test", data=test[0].astype(np.float32), compression="gzip")
		dset = f.create_dataset("Y_test", data=test[1].astype(np.float32), compression="gzip")
	

def dataset_keys_hdf5(file_path):

	dataset = h5py.File(file_path, 'r')
	keys = []
	for key in dataset.keys():
		keys.append(str(key))

	return np.array(keys)


def get_experiments_hdf5(file_path):
	dataset = h5py.File(file_path, 'r')
	return np.array(dataset['experiment'])


def get_file_names(dataset_path):
	file_names = []
	for file_name in os.listdir(dataset_path):
		if os.path.splitext(file_name)[1] == '.h5':
			file_names.append(file_name)
	return file_names


def get_clip_indices(pwm, threshold=0.2, window=2):
	def entropy(p):
		s = 0
		for i in range(len(p)):
			if p[i] > 0:
				s -= p[i]*np.log2(p[i])
		return s

	num_nt, num_seq = pwm.shape
	heights = np.zeros((num_nt, num_seq));
	for i in range(num_seq):
		total_height = (np.log2(num_nt) - entropy(pwm[:, i]));
		heights[:,i] = pwm[:,i]*total_height;

	max_heights = np.max(heights, axis=0)
	index = np.where(max_heights > threshold)[0]

	if index.any():
		start = np.maximum(np.min(index) - window, 0)
		end = np.minimum(np.max(index) + window, num_seq)
	else:
		start = 0
		end = num_seq
	return start, end


def load_coordinates(coordinate_path, name='test'):
	with open(coordinate_path, 'rb') as f:
		train_coords = cPickle.load(f)
		valid_coords = cPickle.load(f)
		test_coords = cPickle.load(f)

	if name == 'train':
		chrom = train_coords[0]
		start = train_coords[1]
		end = train_coords[2]
		strand_info = train_coords[3]
	elif name == 'valid':
		chrom = valid_coords[0]
		start = valid_coords[1]
		end = valid_coords[2]
		strand_info = valid_coords[3]
	elif name == 'test':
		chrom = test_coords[0]
		start = test_coords[1]
		end = test_coords[2]
		strand_info = test_coords[3]
	return chrom, start, end, strand_info




def ensemble_predictions(test, rbp_name, models, input_shape, output_shape, best_path, use_scope=True):
	predictions = []
	for model in models:


		# load model
		genome_model = import_model(model)
		model_layers, optimization = genome_model.model(input_shape, output_shape)

		# build neural network class
		nnmodel = nn.NeuralNet(seed=247)
		nnmodel.build_layers(model_layers, optimization, use_scope=use_scope)

		file_path = os.path.join(best_path, model, rbp_name)
		nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

		# initialize session
		sess = utils.initialize_session(nnmodel.placeholders)

		# load best model
		nntrainer.set_best_parameters(sess, verbose=0)

		predictions.append(nntrainer.get_activations(sess, test))

	predictions = np.hstack(predictions)
	ensemble_predictions = np.mean(predictions,axis=1)
	return ensemble_predictions, predictions


def ensemble_clip_predictions(test, rbp_name, models, input_shape, output_shape, best_path, use_scope=True):
	predictions = []
	for model in models:


		# load model
		genome_model = import_model(model)
		model_layers, optimization = genome_model.model(input_shape, output_shape)

		# build neural network class
		nnmodel = nn.NeuralNet(seed=247)
		nnmodel.build_layers(model_layers, optimization, use_scope=use_scope)

		file_path = os.path.join(best_path, rbp_name)
		nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

		# initialize session
		sess = utils.initialize_session(nnmodel.placeholders)

		# load best model
		nntrainer.set_best_parameters(sess, verbose=0)

		predictions.append(nntrainer.get_activations(sess, test))

	predictions = np.hstack(predictions)
	ensemble_predictions = np.mean(predictions,axis=1)
	return ensemble_predictions, predictions



def ensemble_saliency(X, models, best_path, rbp_name, input_shape, output_shape, use_scope=True):

	guided_saliency = []
	for model in models:
		# load model
		genome_model = import_model(model)
		model_layers, optimization = genome_model.model(input_shape, output_shape)

		# parameters for saliency analysis
		params = {'input_shape': input_shape,
				  'output_shape': output_shape,
				 'genome_model': genome_model.model, 
				 'model_path': os.path.join(best_path, model, rbp_name+'_best.ckpt'),
				 'optimization': optimization,
				 'use_scope': use_scope
				 }				  
		# guided backprop saliency
		guided_saliency.append(saliency.guided_backprop(X, layer='output', class_index=0, params=params))

	for i in range(len(guided_saliency)):
		for j in range(len(models)):
			MAX = np.max(guided_saliency[j][i])
			guided_saliency[j][i] = guided_saliency[j][i]/MAX
	guided_saliency = np.array(guided_saliency)
	mean_saliency = np.mean(guided_saliency, axis=0)
	return mean_saliency, guided_saliency



def ensemble_clip_saliency(X, models, best_path, rbp_name, input_shape, output_shape, use_scope=True):

	guided_saliency = []
	for model in models:
		# load model
		genome_model = import_model(model)
		model_layers, optimization = genome_model.model(input_shape, output_shape)

		# parameters for saliency analysis
		params = {'input_shape': input_shape,
				  'output_shape': output_shape,
				 'genome_model': genome_model.model, 
				 'model_path': os.path.join(best_path, rbp_name+'_best.ckpt'),
				 'optimization': optimization,
				 'use_scope': use_scope
				 }				  
		# guided backprop saliency
		guided_saliency.append(saliency.guided_backprop(X, layer='output', class_index=0, params=params))

	for i in range(len(guided_saliency)):
		for j in range(len(models)):
			MAX = np.max(guided_saliency[j][i])
			guided_saliency[j][i] = guided_saliency[j][i]/MAX
	guided_saliency = np.array(guided_saliency)
	mean_saliency = np.mean(guided_saliency, axis=0)
	return mean_saliency, guided_saliency


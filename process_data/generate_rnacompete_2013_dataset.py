#---------------------------------------------------------------------------------------
"""
Summary: Generates hdf5 file with all 2013 RNAcompete experiments.
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import pandas as pd
import numpy as np
from six.moves import cPickle
np.random.seed(100)

import wrangler

data_path = '../../data/RNAcompete_2013'

# load binding affinities for each rnacompete experiment
df = pd.read_csv(os.path.join(data_path,'targets.tsv'), sep='\t')
experiments = list(df.columns.values)
targets = df.as_matrix()

# load sequences
df = pd.read_csv(os.path.join(data_path,'sequences.tsv'), sep='\t')
rnac_set = df['Fold ID'].as_matrix()
sequences = df['seq'].as_matrix()

# get the maximum length sequence
max_length = 0
for seq in sequences:
    max_length = np.maximum(max_length, len(seq))

# convert sequences into one-hot representation
one_hot = wrangler.munge.convert_one_hot(sequences, max_length)

# save sequences in a fasta format (for rnaplfold)
fasta_path = os.path.join(data_path,'sequences.fa')
wrangler.fasta.generate_fasta(sequences, fasta_path)

# generate secondary structure profiles with rnaplfold
profile_path = os.path.join(data_path,'rnaplfold')
structure = wrangler.structure.RNAplfold_profile(fasta_path, profile_path, window=max_length)

# merge sequences and structural profiles
data = np.concatenate([one_hot, structure], axis=1)

# split dataset into train, cross-validation, and test set
train, valid, test = wrangler.munge.split_rnacompete_dataset(data, targets, rnac_set, valid_frac=0.1)

# save dataset
save_path = os.path.join(data_path, 'rnacompete2013.h5')
print('saving dataset: ', save_path)
with h5py.File(save_path, "w") as f:
    dset = f.create_dataset("X_train", data=train[0].astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_train", data=train[1].astype(np.float32), compression="gzip")
    dset = f.create_dataset("X_valid", data=valid[0].astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_valid", data=valid[1].astype(np.float32), compression="gzip")
    dset = f.create_dataset("X_test", data=test[0].astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_test", data=test[1].astype(np.float32), compression="gzip")
    dset = f.create_dataset("experiment", data=experiments, compression="gzip")

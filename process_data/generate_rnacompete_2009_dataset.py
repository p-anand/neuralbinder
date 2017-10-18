#---------------------------------------------------------------------------------------
"""
Summary: Generates hdf5 file with all 2009 RNAcompete experiments.
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

rbp_names = ['Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1']
data_path = '../../data/RNAcompete_2009'

# save datasets as hdf5 file
save_path = os.path.join(data_path, 'rnacompete2009.h5')
with h5py.File(save_path, "w") as f:
    for rbp_name in rbp_names:

        # process set A dataset
        df = pd.read_csv(os.path.join(data_path,rbp_name+'_data_full_A.txt'), sep='\t', header=None)
        targets_A = np.expand_dims(df[0].as_matrix(), axis=1)
        sequences_A = df[1].as_matrix()

        # get the maximum length sequence
        max_length = 0
        for seq in sequences_A:
            max_length = np.maximum(max_length, len(seq))

        # convert sequences into one-hot representation
        one_hot_A = wrangler.munge.convert_one_hot(sequences_A, max_length)

        # save sequences in a fasta format (for rnaplfold)
        fasta_path = os.path.join(data_path, rbp_name+'_sequences_A.fa')
        wrangler.fasta.generate_fasta(sequences_A, fasta_path)

        # generate secondary structure profiles with rnaplfold
        profile_path = os.path.join(data_path, rbp_name+'_rnaplfold_A')
        structure_A = wrangler.structure.RNAplfold_profile(fasta_path, profile_path, window=max_length)

        # merge sequences and structural profiles
        data_A = np.concatenate([one_hot_A, structure_A], axis=1)

        # process set B dataset
        df = pd.read_csv(os.path.join(data_path,rbp_name+'_data_full_B.txt'), sep='\t', header=None)
        targets_B = np.expand_dims(df[0].as_matrix(), axis=1)
        sequences_B = df[1].as_matrix()

        # convert sequences into one-hot representation
        one_hot_B = wrangler.munge.convert_one_hot(sequences_B, max_length)

        # save sequences in a fasta format (for rnaplfold)
        fasta_path = os.path.join(data_path, rbp_name+'_sequences_B.fa')
        wrangler.fasta.generate_fasta(sequences_B, fasta_path)

        # generate secondary structure profiles with rnaplfold
        profile_path = os.path.join(data_path, rbp_name+'_rnaplfold_B')
        structure_B = wrangler.structure.RNAplfold_profile(fasta_path, profile_path, window=max_length)

        # merge sequences and structural profiles
        data_B = np.concatenate([one_hot_B, structure_B], axis=1)

        # merge set A and set B sequences
        data = np.vstack([data_A, data_B])
        targets = np.vstack([targets_A, targets_B])

        # set up set A and set B labels for each sequence
        experiment = np.concatenate([np.tile('A',len(one_hot_A)), np.tile('B',len(one_hot_B))])

        # split dataset into train, cross-validation, and test set
        train, valid, test = wrangler.munge.split_rnacompete_dataset(data, targets, experiment, valid_frac=0.1)

        # save dataset as a group with the rbp name
        grp = f.create_group(rbp_name)
        dset = grp.create_dataset("X_train", data=train[0].astype(np.float32), compression="gzip")
        dset = grp.create_dataset("Y_train", data=train[1].astype(np.float32), compression="gzip")
        dset = grp.create_dataset("X_valid", data=valid[0].astype(np.float32), compression="gzip")
        dset = grp.create_dataset("Y_valid", data=valid[1].astype(np.float32), compression="gzip")
        dset = grp.create_dataset("X_test", data=test[0].astype(np.float32), compression="gzip")
        dset = grp.create_dataset("Y_test", data=test[1].astype(np.float32), compression="gzip")

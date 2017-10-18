#---------------------------------------------------------------------------------------
"""
Summary: Generate hdf5 files for each eclip experiment for a fixed length window
about each peak.  The background data uses the peaks called for the control
experiments, which we refer to as non-specific.
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
from six.moves import cPickle
import numpy as np
import pandas as pd

import wrangler

np.random.seed(247)

#-----------------------------------------------------------------------------------------------

# fixed_length window to extract about peaks
window = 200

# parent path
data_path = '/media/peter/storage/encode_eclip'

# path to peaks (after running process_peaks.py)
peak_path = os.path.join(data_path, 'peaks', 'yeo_peaks')

# path to background peaks (after running process_background_peaks.py)
background_path = os.path.join(data_path, 'peaks', 'background_peaks')

# path to chromosome sizes
chrom_path = os.path.join(data_path, 'chrom_sizes.bed')

# path to save eclip datasets
dataset_path = wrangler.utils.make_directory(data_path, 'eclip_datasets')

#-----------------------------------------------------------------------------------------------
# preliminaries

# parse meta data file
parse_list = ['File accession', 'Experiment accession',
              'Biological replicate(s)', 'Experiment target', 'Biosample term name']
meta_data_path = os.path.join(data_path, 'eclip_bed_data', 'metadata.tsv')
values = wrangler.encode.parse_metadata(meta_data_path, parse_list, genome='GRCh38', file_format='bed narrowPeak')

# get values from dictionary
file_names = values['File accession']
experiments = values['Experiment accession']
replicates = values['Biological replicate(s)']
targets = values['Experiment target']
cell_types = values['Biosample term name']

# set genome path
genome_path = os.path.join(data_path, 'hg38.fa')
if not os.path.isfile(genome_path):
    wrangler.download.reference_genome('hg38', data_path)

#-----------------------------------------------------------------------------------------------
# process data

# loop over each experiment and process replicates
unique_experiments = np.unique(experiments)
for experiment in unique_experiments:

    index = np.where(experiments == experiment)[0]
    rbp_name = targets[index[0]][:targets[index[0]].index('-')]
    cell_name = cell_types[index[0]]
    experiment_name = rbp_name + '_' + cell_name
    print('processing: ', experiment_name)

    # make directory to save temporary files
    experiment_path = os.path.join(peak_path, experiment_name)

    #-----------------------------------------------------------------------------------------------
    # process positive control sequences

    # path to positive control peak
    overlap_path = os.path.join(experiment_path, experiment_name+'_overlap.bed')

    # enforce constant size
    print('extracting positive sequences')
    overlap_window_path = os.path.join(experiment_path, experiment_name+'_overlap_'+str(window)+'.bed')
    wrangler.bedtools.enforce_constant_size(overlap_path, overlap_window_path, window)

    # get sequences from bed file and save as fasta file
    overlap_fasta_path = os.path.join(experiment_path, experiment_name+'_overlap_'+str(window)+'.fa')
    wrangler.bedtools.to_fasta(overlap_window_path, overlap_fasta_path, genome_path)

    # parse sequence and chromosome from fasta file
    pos_sequences = wrangler.fasta.parse_sequences(overlap_fasta_path)

    # filter sequences with absent nucleotides
    pos_sequences_filtered, pos_good_index = wrangler.munge.filter_nonsense_sequences(pos_sequences)

    # convert filtered sequences to one-hot representation
    pos_one_hot = wrangler.munge.convert_one_hot(pos_sequences_filtered, max_length=window)

    # generate fasta of sequences
    print('calculating secondary structure for positive sequences')
    fasta_path = os.path.join(experiment_path, experiment_name+'_overlap_'+str(window)+'_filtered.fa')
    wrangler.fasta.generate_fasta(pos_sequences_filtered, fasta_path)

    # predict secondary structure_path
    structure_path = os.path.join(experiment_path, 'pos_'+str(window)+'_')
    pos_structure = wrangler.structure.RNAplfold_profile(fasta_path, structure_path, window=window)

    # concatenate sequence, structure, and conservation
    pos_one_hot_all = np.concatenate([pos_one_hot, pos_structure], axis=1)


    #-----------------------------------------------------------------------------------------------
    # process negative control sequences

    # enforce constant window size for each replicate
    print('processing negative sequences')
    rep1_sort_path = os.path.join(experiment_path, experiment_name+'_rep1_sorted.bed')
    rep1_window_path = os.path.join(experiment_path, experiment_name+'_rep1_sorted_'+str(window)+'.bed')
    wrangler.bedtools.enforce_constant_size(rep1_sort_path, rep1_window_path, window)

    rep2_sort_path = os.path.join(experiment_path, experiment_name+'_rep2_sorted.bed')
    rep2_window_path = os.path.join(experiment_path, experiment_name+'_rep2_sorted_'+str(window)+'.bed')
    wrangler.bedtools.enforce_constant_size(rep2_sort_path, rep2_window_path, window)

    # enforce constant window size for background bed file
    background_file_path = os.path.join(background_path, experiment_name+'.bed')
    background_window_path = os.path.join(background_path, experiment_name+'_'+str(window)+'.bed')
    wrangler.bedtools.enforce_constant_size(background_file_path, background_window_path, window)

    # find background peaks that do not overlap with any replicates
    background_unsorted_path = os.path.join(experiment_path, experiment_name+'_background_unsorted.bed')
    wrangler.bedtools.nonoverlap(background_window_path, [rep1_window_path, rep2_window_path], background_unsorted_path, options=['-wa'])

    # sort background peaks
    background_sort_path = os.path.join(experiment_path, experiment_name+'_background_sorted.bed')
    wrangler.bedtools.sort(background_unsorted_path, background_sort_path, verbose=1)

    # merge redundant background entries
    background_merge_path = os.path.join(experiment_path, experiment_name+'_background.bed')
    wrangler.bedtools.merge(background_sort_path, background_merge_path)

    # reinforce constant size after merge
    print('extracting negative sequences')
    background_window_path = os.path.join(experiment_path, experiment_name+'_background_'+str(window)+'.bed')
    wrangler.bedtools.enforce_constant_size(background_merge_path, background_window_path, window)

    # get sequences from bed file and save as fasta file
    background_fasta_path = os.path.join(experiment_path, experiment_name+'_background_'+str(window)+'.fa')
    wrangler.bedtools.to_fasta(background_window_path, background_fasta_path, genome_path)

    # parse sequence and chromosome from fasta file
    neg_sequences = wrangler.fasta.parse_sequences(background_fasta_path)

    # filter sequences with non-nucleotides, i.e. N
    neg_sequences_filtered, neg_good_index = wrangler.munge.filter_nonsense_sequences(neg_sequences)

    # convert bacgkround sequence to one-hot representation
    shuffle = np.random.permutation(len(neg_sequences_filtered))[:len(pos_sequences_filtered)]
    neg_one_hot = wrangler.munge.convert_one_hot(neg_sequences_filtered[shuffle], max_length=window)
    neg_good_index = neg_good_index[shuffle]

    # save filtered sequences to fasta file
    print('calculating secondary structure for negative sequences')
    fasta_path = os.path.join(experiment_path, experiment_name+'_overlap_'+'negative_'+str(window)+'_filtered.fa')
    wrangler.fasta.generate_fasta(neg_sequences_filtered[shuffle], fasta_path)

    # predict secondary structure
    structure_path = os.path.join(experiment_path, 'neg_'+str(window)+'_')
    neg_structure = wrangler.structure.RNAplfold_profile(fasta_path, structure_path, window=window)

    # concatenate sequence, structure, and conservation
    neg_one_hot_all = np.concatenate([neg_one_hot, neg_structure], axis=1)

    #-----------------------------------------------------------------------------------------------
    # merge postive and negative sequences

    print('saving dataset')
    one_hot = np.vstack([pos_one_hot_all, neg_one_hot_all])
    labels = np.vstack([np.ones((len(pos_one_hot_all), 1)), np.zeros((len(neg_one_hot_all), 1))])

    # split dataset into training set, cross-validation set, and test set
    train, valid, test, indices = wrangler.munge.split_dataset(one_hot, labels, valid_frac=0.1, test_frac=0.2)

    # save dataset to hdf5 file
    with h5py.File(os.path.join(dataset_path, experiment_name+'_'+str(window)+'.h5'), 'w') as fout:
        X_train = fout.create_dataset('X_train', data=train[0], dtype='float32', compression="gzip")
        Y_train = fout.create_dataset('Y_train', data=train[1], dtype='int8', compression="gzip")
        X_valid = fout.create_dataset('X_valid', data=valid[0], dtype='float32', compression="gzip")
        Y_valid = fout.create_dataset('Y_valid', data=valid[1], dtype='int8', compression="gzip")
        X_test = fout.create_dataset('X_test', data=test[0], dtype='float32', compression="gzip")
        Y_test = fout.create_dataset('Y_test', data=test[1], dtype='int8', compression="gzip")

    #-----------------------------------------------------------------------------------------------
    # save coordinates of each sequence

    df = pd.read_csv(overlap_window_path, sep='\t', header=None)
    chrom = df[0].as_matrix()
    start = df[1].as_matrix()
    end = df[2].as_matrix()
    strand = df[5].as_matrix()

    df2 = pd.read_csv(background_window_path, sep='\t', header=None)
    chrom2 = df2[0].as_matrix()
    start2 = df2[1].as_matrix()
    end2 = df2[2].as_matrix()
    strand2 = df2[5].as_matrix()

    chrom_all = np.concatenate([chrom[pos_good_index], chrom2[neg_good_index]])
    start_all = np.concatenate([start[pos_good_index], start2[neg_good_index]])
    end_all = np.concatenate([end[pos_good_index], end2[neg_good_index]])
    strand_all = np.concatenate([strand[pos_good_index], strand2[neg_good_index]])

    train_coordinates = (chrom_all[indices[0]], start_all[indices[0]], end_all[indices[0]], strand_all[indices[0]])
    valid_coordinates = (chrom_all[indices[1]], start_all[indices[1]], end_all[indices[1]], strand_all[indices[1]])
    test_coordinates = (chrom_all[indices[2]], start_all[indices[2]], end_all[indices[2]], strand_all[indices[2]])
    with open(os.path.join(data_path, experiment_name+'_'+str(window)+'_coordinates.pickle'), 'wb') as f_save:
        cPickle.dump(train_coordinates, f_save, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(valid_coordinates, f_save, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_coordinates, f_save, protocol=cPickle.HIGHEST_PROTOCOL)

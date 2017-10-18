#---------------------------------------------------------------------------------------
"""
Summary: Process peaks called by CLIPPER (Yeo lab) to find
conservative peaks across replicates.
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from six.moves import cPickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wrangler

#-----------------------------------------------------------------------------------------------

# thresholds for peaks
signal_thresh = 1
p_value_thresh = 3

# parent path
data_path = '/media/peter/storage/encode_eclip'

# bed file path
bed_path = os.path.join(data_path, 'eclip_bed_data')

# path to save new bed files
peak_path = wrangler.utils.make_directory(data_path, 'peaks')
bed_save_path = wrangler.utils.make_directory(peak_path, 'yeo_peaks')

# bath to save statistics
statistics_path = wrangler.utils.make_directory(bed_save_path, 'statistics')

#-----------------------------------------------------------------------------------------------
# process data

# parse meta data file
parse_list = ['File accession', 'Experiment accession',
              'Biological replicate(s)', 'Experiment target', 'Biosample term name']
meta_data_path = os.path.join(bed_path, 'metadata.tsv')
values = wrangler.encode.parse_metadata(meta_data_path, parse_list, genome='GRCh38', file_format='bed narrowPeak')

# get values from dictionary
file_names = values['File accession']
experiments = values['Experiment accession']
replicates = values['Biological replicate(s)']
targets = values['Experiment target']
cell_types = values['Biosample term name']

# statistics file to save info
statistics_file_path = os.path.join(statistics_path, 'statistics.txt')
f = open(statistics_file_path, 'w')

# get unique experiments
unique_experiments = np.unique(experiments)

# loop over each experiment and process replicates
for experiment in unique_experiments:

	# find replicate experiments
	index = np.where(experiments == experiment)[0]
	rep1_index = index[0]
	rep2_index = index[1]

	# get rbp and cell type
	rbp_name = targets[rep1_index][:targets[rep1_index].index('-')]
	cell_name = cell_types[rep1_index]
	experiment_name = rbp_name + '_' + cell_name
	rep1_name = experiment_name+'_rep'+str(replicates[rep1_index])
	rep2_name = experiment_name+'_rep'+str(replicates[rep2_index])

	# make directory to save temporary files
	tmp_path = wrangler.utils.make_directory(bed_save_path, experiment_name)

	#-----------------------------------------------------------------------------------------------
	# process replicates

	# sort overlap files for each replicate
	rep1_path = os.path.join(bed_path, file_names[rep1_index]+'.bed.gz')
	rep1_sort_path = os.path.join(tmp_path, rep1_name +'_sorted.bed')
	wrangler.bedtools.sort(rep1_path, rep1_sort_path, verbose=1)

	rep2_path = os.path.join(bed_path, file_names[rep2_index]+'.bed.gz')
	rep2_sort_path = os.path.join(tmp_path, rep2_name+'_sorted.bed')
	wrangler.bedtools.sort(rep2_path, rep2_sort_path, verbose=1)

	#-----------------------------------------------------------------------------------------------
	# merge unfiltered replicates

	# overlap between unfiltered replicates
	overlap_unfiltered_unsorted_path = os.path.join(tmp_path, experiment_name+'_overlap_unfiltered_unsorted.bed')
	wrangler.bedtools.overlap(rep1_sort_path, rep2_sort_path, overlap_unfiltered_unsorted_path, options=['-wa', '-wb', '-s'], verbose=1)

	# sort overlap files
	overlap_unfiltered_sorted_path = os.path.join(tmp_path, experiment_name+'_overlap_unfiltered_sorted.bed')
	wrangler.bedtools.sort(overlap_unfiltered_unsorted_path, overlap_unfiltered_sorted_path, verbose=1)

	# merge redundant entries
	merge_unfiltered_path = os.path.join(tmp_path, experiment_name+'_overlap_unfiltered.bed')
	wrangler.bedtools.merge(overlap_unfiltered_sorted_path, merge_unfiltered_path)


	#-----------------------------------------------------------------------------------------------
	# filter replicates

	# filter peaks with low signal values
	rep1_signal_filter_path = os.path.join(tmp_path, rep1_name+'_signal.bed')
	wrangler.bash.positive_filter(rep1_sort_path, rep1_signal_filter_path,
	                              signal=7, threshold=signal_thresh, verbose=1)

	rep2_signal_filter_path = os.path.join(tmp_path, rep2_name+'_signal.bed')
	wrangler.bash.positive_filter(rep2_sort_path, rep2_signal_filter_path,
	                              signal=7, threshold=signal_thresh, verbose=1)

	# filter peaks low p-values
	rep1_pvalue_filter_path = os.path.join(tmp_path, rep1_name+'_pvalue.bed')
	wrangler.bash.positive_filter(rep1_signal_filter_path, rep1_pvalue_filter_path,
	                              signal=8, threshold=p_value_thresh, verbose=1)

	rep2_pvalue_filter_path = os.path.join(tmp_path, rep2_name+'_pvalue.bed')
	wrangler.bash.positive_filter(rep2_signal_filter_path, rep2_pvalue_filter_path,
	                              signal=8, threshold=p_value_thresh, verbose=1)

	#-----------------------------------------------------------------------------------------------
	# merge filtered replicates

	# overlap between filtered replicates
	overlap_unsorted_path = os.path.join(tmp_path, experiment_name+'_overlap_unsorted.bed')
	wrangler.bedtools.overlap(rep1_pvalue_filter_path, rep2_pvalue_filter_path, overlap_unsorted_path, options=['-wa', '-wb', '-s'], verbose=1)

	# sort overlap files
	overlap_sort_path = os.path.join(tmp_path, experiment_name+'_overlap_sorted.bed')
	wrangler.bedtools.sort(overlap_unsorted_path, overlap_sort_path, verbose=1)

	# filter off-diagonal signal values across replicates
	correlation_path = os.path.join(tmp_path, experiment_name+'_correlation.bed')
	wrangler.bash.correlation_filter(overlap_sort_path, correlation_path, signals=[7, 17], equality='<', threshold=1.0, verbose=1)

	# merge redundant entries
	merge_path = os.path.join(tmp_path, experiment_name+'_overlap.bed')
	wrangler.bedtools.merge(correlation_path, merge_path)


	#-----------------------------------------------------------------------------------------------
	# plot scatter between replicates

	fig = plt.figure()

	# scatter plot of signal values between unfiltered replicates that overlap
	df = pd.read_csv(overlap_unfiltered_sorted_path, sep='\t', header=None)
	start = df[1].as_matrix()
	end = df[2].as_matrix()
	rep1 = df[6].as_matrix()
	p1 = df[7].as_matrix()
	rep2 = df[16].as_matrix()
	p2 = df[17].as_matrix()
	num_overlap_total = len(rep1)
	corr_overlap_total = np.corrcoef(rep1, rep2)[0][1]
	plt.scatter(rep1, rep2, c='k')

	# scatter plot of signal values between filtered replicates that overlap
	df = pd.read_csv(overlap_sort_path, sep='\t', header=None)
	start = df[1].as_matrix()
	end = df[2].as_matrix()
	rep1 = df[6].as_matrix()
	p1 = df[7].as_matrix()
	rep2 = df[16].as_matrix()
	p2 = df[17].as_matrix()
	num_overlap_filtered = len(rep1)
	corr_overlap_filtered = np.corrcoef(rep1, rep2)[0][1]
	plt.scatter(rep1, rep2, c='r')

	# scatter plot of signal values between correlation filtered replicates that overlap
	df = pd.read_csv(correlation_path, sep='\t', header=None)
	start = df[1].as_matrix()
	end = df[2].as_matrix()
	rep1 = df[6].as_matrix()
	p1 = df[7].as_matrix()
	rep2 = df[16].as_matrix()
	p2 = df[17].as_matrix()
	num_overlap = len(rep1)
	corr_overlap = np.corrcoef(rep1, rep2)[0][1]

	plt.scatter(rep1, rep2)
	plt.xlabel('Signal value (rep 1)', fontsize=14)
	plt.ylabel('Signal value (rep 2)', fontsize=14)
	outfile = os.path.join(statistics_path, experiment_name+'_scatter')
	fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

	fig = plt.figure()
	size = end-start
	plt.hist(size, bins=50);
	plt.xlabel('Merged coordinate size (nt)', fontsize=14)
	plt.ylabel('Counts', fontsize=14)
	outfile = os.path.join(statistics_path, experiment_name+'_size')
	fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

	df = pd.read_csv(rep1_path, sep='\t', header=None)
	start = df[1].as_matrix()
	end = df[2].as_matrix()
	num_rep1 = len(df[1].as_matrix())

	df = pd.read_csv(rep2_path, sep='\t', header=None)
	start = df[1].as_matrix()
	end = df[2].as_matrix()
	num_rep2 = len(df[1].as_matrix())

	# append statistics to file
	f.write('%s\t%s\t%d\t%d\t%d\t%.4f\t%d\t%.4f\t%d\t%.4f\n' % (cell_name, rbp_name, num_rep1, num_rep2, num_overlap_total, corr_overlap_total, num_overlap_filtered, corr_overlap_filtered, num_overlap, corr_overlap))
f.close()

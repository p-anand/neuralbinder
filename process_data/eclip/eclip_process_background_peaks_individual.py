#---------------------------------------------------------------------------------------
"""
Summary: Use Piranha to call peaks in for a subset of control experiments
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np

sys.path.append('..')
import wrangler

#---------------------------------------------------------------------------------------------------

# peak calling parameters
bin_size = 20
p_value = 0.01

# paths
data_path = '/media/peter/storage/encode_eclip'
bam_path = os.path.join(data_path, 'eclip_bam_controls')

# output directory
peak_path = wrangler.utils.make_directory(data_path, 'peaks')
save_path = wrangler.utils.make_directory(peak_path, 'background_peaks')

#---------------------------------------------------------------------------------------------------

# parse metadata
parse_list = ['File accession', 'Experiment accession',
              'Biological replicate(s)', 'Experiment target', 'Biosample term name']
meta_data_path = os.path.join(bam_path, 'metadata.tsv')
values = wrangler.encode.parse_metadata(meta_data_path, parse_list, genome='GRCh38', file_format='bam')

# get values from dictionary
file_names = values['File accession']
experiments = values['Experiment accession']
replicates = values['Biological replicate(s)']
targets = values['Experiment target']
cell_types = values['Biosample term name']

files = ['ENCFF568BOI', 'ENCFF895BNM', 'ENCFF735PKL', 'ENCFF269WSV', 'ENCFF791AGL', 'ENCFF463LHG']
#          HLTF_K562      PTBP1-HepG2     SF3B K562      TNRC6A_K562   PTBP1-K562    DDX6-HepG2
for j in range(len(files)):
    i = np.where(files[j] == file_names)[0][0]
    print(i)
    print('Peak calling: ' + targets[i] + ' ' + cell_types[i])
    rbp_name = targets[i][:targets[i].index(' ')]
    experiment_name = rbp_name + '_' + cell_types[i]

    file_path = os.path.join(bam_path, file_names[i]+'.bam')
    output_path = os.path.join(save_path, experiment_name+'.bed')
    wrangler.peaks.call_peaks(file_path, output_path, bin_size=bin_size, p_value=p_value)

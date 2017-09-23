from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, sys, h5py
import numpy as np
from six.moves import cPickle

sys.path.append('..')
import wrangler

sys.path.append('../../deepomics')
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.misc import imresize



def load_coordinates(coordinate_path, name='test'):

    with open(coordinate_path, 'rb') as f_save:
        chrom = cPickle.load(f_save)
        start = cPickle.load(f_save)
        end = cPickle.load(f_save)
        strand = cPickle.load(f_save)
        indices = cPickle.load(f_save)

    if name == 'train':
        index = 0
    elif name == 'valid':
        index = 1
    elif name == 'test':
        index = 2

    chrom = chrom[indices[index]]
    start = start[indices[index]]
    end = end[indices[index]]
    strand = strand[indices[index]]
    return chrom, start, end, strand


def plot_top_saliency(sess, nntrainer, test, params, save_path, coordinate_path, name='test', num_plots=10):

    # get coordinates of sequences
    chrom, start, end, strand = load_coordinates(coordinate_path, name='test')

    # plot top 20 predictions
    predictions = nntrainer.get_activations(sess, test, 'output')
    true_index = np.where(test['targets'][:,0] == 1)[0]
    max_index = true_index[np.argsort(predictions[true_index,0])[::-1]]

    # plot saliency
    num_plots = 20
    for index in max_index[:num_plots]:
        X = np.expand_dims(test['inputs'][index], axis=0)

        # guided backprop saliency
        guided_saliency = saliency.guided_backprop(X, layer='output', class_index=0, params=params)

        # stochastic guided backprop saliency
        #guided_saliency = saliency.stochastic_guided_backprop(X, layer='output', class_index=None, params=params,
        #                                                      num_average=200, threshold=0.0, stochastic_val=0.5)

        # plot saliency comparison
        fig, plt = visualize.plot_seq_neg_saliency(np.squeeze(X).T, np.squeeze(guided_saliency[0]).T, figsize=(100,16), alphabet='rna')

        if strand[index] == '+':
            strand_info = 'pos'
        else:
            strand_info = 'neg'

        outfile = os.path.join(save_path, str(chrom[index])+':'+str(start[index])+'-'+str(end[index])+'_'+strand_info+'.pdf')
        fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight') 

        plt.close()



def convert_one_hot_filter(train):
    train_sequences = wrangler.munge.convert_one_hot_sequences(np.squeeze(train['inputs']))
    seq_length = train['inputs'].shape[1]
    index = []
    for i, seq in enumerate(train_sequences):
        if len(seq) == seq_length:
            index.append(i)
    return np.array(train_sequences)[index], train['targets'][index]






def filter_canonical_chromosomes(bed_path, output_path):

    chrom_sizes = { 'chr1': 248956422,
                    'chr2': 242193529,
                    'chr3': 198295559,
                    'chr4': 190214555,
                    'chr5': 181538259,
                    'chr6': 170805979,
                    'chr7': 159345973,
                    'chr8': 145138636,
                    'chr9': 138394717,
                    'chr11': 135086622,
                    'chr10': 133797422,
                    'chr12': 133275309,
                    'chr13': 114364328,
                    'chr14': 107043718,
                    'chr15': 101991189,
                    'chr16': 90338345,
                    'chr17': 83257441,
                    'chr18': 80373285,
                    'chr20': 64444167,
                    'chr19': 58617616,
                    'chr22': 50818468,
                    'chr21': 46709983,
                    }

    with open(output_path, "w") as fout:
        with open(bed_path, "r") as fin:
            for line in fin:
                chrom, start, end, name, length, strand = line.split()
                if str(chrom) in chrom_sizes:
                    fout.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (chrom, start, end, name, length, strand))







def plot_top_pos_saliency(sess, nntrainer, test, params, save_path, coordinate_path, name='test', num_plots=10, struct=None, conservation=None, figsize=(120,25)):

    # get coordinates of sequences
    chrom, start, end, strand = load_coordinates(coordinate_path, name='test')

    # plot top 20 predictions
    predictions = nntrainer.get_activations(sess, test, 'output')
    true_index = np.where(test['targets'][:,0] == 1)[0]
    max_index = true_index[np.argsort(predictions[true_index,0])[::-1]]

    # plot saliency
    for index in max_index[:num_plots]:
        X = np.expand_dims(test['inputs'][index], axis=0)

        # guided backprop saliency
        guided_saliency = saliency.guided_backprop(X, layer='output', class_index=0, params=params)

        # plot saliency comparison
        fig, plt = plot_seq_pos_saliency(X, guided_saliency, struct, conservation, figsize=figsize)

        if strand[index] == '+':
            strand_info = 'pos'
        else:
            strand_info = 'neg'

        outfile = os.path.join(save_path, str(chrom[index])+':'+str(start[index])+'-'+str(end[index])+'_'+strand_info+'.pdf')
        fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight') 

        plt.close()





def plot_seq_pos_saliency(X, saliency, struct, conservation, figsize=(100,8), factor=3):
    
    W = np.squeeze(saliency[0]).T
    X = np.squeeze(X).T
    
    height=50
    nt_width=20
    num_rows = 2
        
    if struct and conservation:
        num_rows += 4
    elif struct and not conservation:
        num_rows += 2        
    elif not struct and conservation:
        num_rows += 1        

    grid = mpl.gridspec.GridSpec(num_rows, 1)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2) 

    fig = plt.figure(figsize=figsize);

    subfig = plt.subplot(grid[0])
    pwm = utils.normalize_pwm(W[:4,:], factor=factor)
    pos_logo = visualize.seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet='rna')
    plt.imshow(pos_logo, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    subfig.axis('tight')

    subfig = plt.subplot(grid[1])
    logo = visualize.seq_logo(X[:4,:], height=height, nt_width=nt_width, norm=0, alphabet='rna')
    plt.imshow(logo, interpolation='none');
    plt.axis('off');
    subfig.axis('tight')

    if struct and conservation:

        subfig = plt.subplot(grid[2])
        pwm = utils.normalize_pwm(W[4:6,:], factor=factor)
        pos_logo = visualize.seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet='pu')
        plt.imshow(pos_logo, interpolation='none')
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        subfig.axis('tight')

        subfig = plt.subplot(grid[3])
        logo = visualize.seq_logo(X[4:6,:], height=height, nt_width=nt_width, norm=0, alphabet='pu')
        plt.imshow(logo, interpolation='none');
        plt.axis('off');
        subfig.axis('tight')

        subfig =  plt.subplot(grid[4])
        plt.plot(W[6:,:].T, '--');
        plt.plot(X[6:,:].T);
        plt.axis('tight')
        plt.xlim([0,X.shape[0]])
        subfig.axis('tight')


    elif struct and not conservation:

        subfig = plt.subplot(grid[2])
        pwm = utils.normalize_pwm(W[4:6,:], factor=factor)
        pos_logo = visualize.seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet='pu')
        plt.imshow(pos_logo, interpolation='none')
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        subfig.axis('tight')

        subfig = plt.subplot(grid[3])
        logo = visualize.seq_logo(X[4:6,:], height=height, nt_width=nt_width, norm=0, alphabet='pu')
        plt.imshow(logo, interpolation='none');
        plt.xlim([0,X.shape[0]])
        plt.axis('off');
        subfig.axis('tight')

    elif not struct and conservation:

        subfig = plt.subplot(grid[2])
        plt.plot(W[4:,:].T, '--');
        plt.plot(X[4:,:].T);
        plt.xlim([0,X.shape[0]])
        plt.axis('tight')
        subfig.axis('tight')
        
    return fig, plt









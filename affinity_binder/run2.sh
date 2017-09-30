#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python train_rnacompete_2009.py
CUDA_VISIBLE_DEVICES=0 python plot_rnacompete_2009_motifs.py
CUDA_VISIBLE_DEVICES=0 python plot_rnacompete_2009_saliency.py

#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_rnacompete_2013.py
CUDA_VISIBLE_DEVICES=1 python validation_rnacompete_2013.py
CUDA_VISIBLE_DEVICES=1 python test_rnacompete_2013.py

CUDA_VISIBLE_DEVICES=1 python plot_rnacompete_2013_saliency.py
CUDA_VISIBLE_DEVICES=1 python plot_rnacompete_2013_motifs.py
CUDA_VISIBLE_DEVICES=1 python plot_rnacompete_2013_ensemble_motifs.py

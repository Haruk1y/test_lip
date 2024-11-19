#!/bin/sh
#SBATCH -p p
#SBATCH --mem 32GB --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train.py --data_dir data/processed --video_dir data/processed/train/videos --lab_dir data/processed/train/lab --landmark_dir data/processed/train/landmarks --batch_size 8 --epochs 5 --learning_rate 1e-4 --num_workers 2 --checkpoint_dir ../checkpoints --log_dir ../logs
#!/bin/bash
#SBATCH --job-name=yolov8-train
#SBATCH --output=logs/yolov8_%j.out
#SBATCH --error=logs/yolov8_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# Ativa o ambiente Python (ajusta conforme o teu setup)
module load python
source ~/venvs/yolov8/bin/activate  # ou conda activate yolov8

# Corre o script Python
python train.py

#!/bin/bash

#SBATCH --job-name=unzipCOCO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3139@nyu.edu
#SBATCH --output=slurm.out
#SBATCH --gres=gpu

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif
file_name=$1

# start running
singularity exec --nv \
            --overlay ${ext3_path}:ro \
            ${sif_path} /bin/bash -c "source ~/.bashrc
            conda deactivate
            conda activate dl 
            cd /scratch/$USER/Deformable-DETR/data/coco
            unzip ${file_name} | tqdm > /dev/null"
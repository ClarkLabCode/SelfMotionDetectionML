#!/bin/bash
#SBATCH --job-name=motion_data
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda
source activate py3_pytorch

/usr/bin/time python3 select_natural_scenes.py
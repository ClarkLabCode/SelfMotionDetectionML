#!/bin/bash
#SBATCH --job-name=motion_data_object
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda
source activate py3_pytorch

/usr/bin/time python3 get_train_test_data_all_in_one_object.py
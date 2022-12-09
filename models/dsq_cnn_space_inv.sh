#!/bin/bash

module load dSQ

dsq --job-file=joblist_cnn_space_inv.txt \
--job-name=cnn_space_inv \
--partition=day \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=5 \
--mem-per-cpu=15G \
--time=1- \
--mail-type=ALL \
--mail-user=baohua.zhou@yale.edu \
--status-dir=status_files \
--output=dsq_output/dsq-joblist_cnn_space_inv-%A_%a-%N.out
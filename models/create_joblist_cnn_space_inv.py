#!/usr/bin/env python

####### hyperparameter screening wide field, cnn plus dense, with space invariance and left-right symmetry #######

from absl import app
from absl import flags
from ml_collections import config_flags

_FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'configuration file', lock_config=True)
flags.mark_flags_as_required(['config'])

def main(_):
    # get lists of hyper parameters
    _VALUE = _FLAGS.config
    L = _VALUE.L # width of the input array
    D_cnn_list = _VALUE.D_cnn_list # list of depth
    C_list = _VALUE.C_list # list of number of independent channels
    k_list = _VALUE.k_list # list of kernel size
    Repeat = _VALUE.Repeat # repeat for different initializations

    # Save info to a joblist file
    log_file = 'joblist_cnn_space_inv.txt'
    with open(log_file, 'w') as f:
        f.truncate()
        for R in range(Repeat):
            for k in k_list:
                for D_cnn in D_cnn_list:
                    for C in C_list:
                        f.write(f'module load miniconda; source activate py3_pytorch; python3 train_validate_run_cnn_space_inv.py --config=configs_cnn_space_inv.py --D_cnn={D_cnn} --C={C} --k={k} --R={R+1}\n')
                            
if __name__ == '__main__':
  app.run(main)


























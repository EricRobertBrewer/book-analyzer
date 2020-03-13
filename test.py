import os
import sys
import time

import tensorflow as tf
import h5py
import imblearn
import keras
import numpy as np
import pandas
import sklearn


def main():
    print('sys={}'.format(sys.version))
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
        print('SLURM_JOB_ID: {:d}'.format(stamp))
    else:
        stamp = int(time.time())
        print('Time stamp (s): {:d}'.format(stamp))
    print()

    print('h5py={}'.format(h5py.__version__))
    print('imblearn={}'.format(imblearn.__version__))
    print('keras={}'.format(keras.__version__))
    print('numpy={}'.format(np.__version__))
    print('pandas={}'.format(pandas.__version__))
    print('sklearn={}'.format(sklearn.__version__))
    print('tensorflow={}'.format(tf.__version__))
    print('tensorflow.test.is_gpu_available()={}'.format(tf.test.is_gpu_available()))
    print('tensorflow.test.is_gpu_available(cuda_only=True)={}'.format(tf.test.is_gpu_available(cuda_only=True)))
    print('tensorflow.keras={}'.format(tf.keras.__version__))


if __name__ == '__main__':
    main()

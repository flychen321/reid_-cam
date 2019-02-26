import os
import numpy as np

value = [[0, 100], [70, 90], [60, 80], [60, 90], [50, 100], [80, 90], [70, 80], [80, 80], [90, 90], ]
for i in np.arange(1):
    print('i = %.3f' % i)
    log_name = 'log/' + 'log_' + str(i)
    print('log name = %s' % log_name)
    cmd = 'python train_baseline.py --use_dense --data_dir duke --modelname ' + 'model_' + str(i) + ' >> ' + log_name
    print('cmd = %s' % cmd)
    os.system(cmd)

    os.system('python test.py  --use_dense  --mode ' + str(0) + ' >>  ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)

    os.system('python test.py  --use_dense  --mode ' + str(1) + ' >>  ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)

    os.system('python test.py  --use_dense  --mode ' + str(2) + ' >>  ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)

    os.system('python test.py  --use_dense  --mode ' + str(3) + ' >>  ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)

# python train_baseline.py  --use_dense  --data_dir duke  --use_mid  --use_trainall  --modelname  model_0
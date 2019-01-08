from __future__ import print_function, division
import numpy as np
import matplotlib

matplotlib.use('agg')
from PIL import Image
from scipy.io import loadmat
from scipy.io import savemat
import os
dir_path = 'pytorch_result.mat'
features = loadmat(dir_path)
print(features.keys())
print(features['query_f'].shape)
print(features['query_label'].shape)
print(features['query_cam'].shape)
print(features['query_files'].shape)
query_f = features['query_f']
query_label = features['query_label'][0]
query_cam = features['query_cam'][0]
query_files = features['query_files']

cnt_same = 0
cnt_diff = 0
dist_same = 0
dist_diff = 0
same_d = []
diff_d = []

process_num = len(query_label)
# process_num = 500
for i in range(process_num):
    if i % 100 == 0:
        print('i = %4d' % i)
    for j in range(process_num):
        if i != j:
        # if True:
            if query_label[i] == query_label[j]:
                # dist = np.dot(query_f[i], query_f[j])
                dist = np.sqrt(np.sum(np.square(query_f[i] - query_f[j])))
                dist_same += dist
                same_d.append(dist)
                cnt_same += 1
            else:
                # dist = np.dot(query_f[i], query_f[j])
                dist = np.sqrt(np.sum(np.square(query_f[i] - query_f[j])))
                dist_diff += dist
                diff_d.append(dist)
                cnt_diff += 1

print('cnt_same = %d' % cnt_same)
print('cnt_diff = %d' % cnt_diff)
print('dist_same = %.5f' % dist_same)
print('dist_diff = %.5f' % dist_diff)
print('min same_d = %.4f    max same_d = %.4f' % (np.min(same_d), np.max(same_d)))
print('min diff_d = %.4f    max diff_d = %.4f' % (np.min(diff_d), np.max(diff_d)))
print('avg_dist_same = %.5f' % (dist_same/cnt_same))
print('avg_dist_diff = %.5f' % (dist_diff/cnt_diff))





# -*- coding: utf-8 -*-
'''
if the model is trained by multi-GPU,  use the upper load_network() function, else use the load_network() below.
'''
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense
from PIL import Image
import torch.nn.functional as F
import glob

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='best', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_DesNet121', type=str, help='save model path')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--ratio', default=65, type=str, help='ratio')

opt = parser.parse_args()
opt.use_dense = True
print('ratio = %.3f' % (float(opt.ratio)/100.0))
#str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        # transforms.Resize((288,144), interpolation=3),
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()

######################################################################
# Load model

#----------single gpu training-----------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    # print(network.model.features.conv0.weight[0][0])

    torch.save(network, os.path.join('./model',name,'whole_net_%s.pth'%opt.which_epoch))
    torch.save(network.state_dict(), os.path.join('./model',name,'param_net_%s.pth'%opt.which_epoch))
    recovery_net = torch.load(os.path.join('./model',name,'whole_net_%s.pth'%'best'))
    # print(recovery_net.model.features.conv0.weight[0][0])
    # exit()
    return network


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_one_feature(path, model):
    features = torch.FloatTensor()
    input_image = Image.open(path)
    file = os.path.split(path)[-1]
    label = int(file[6]) - 1
    input_image = data_transforms(input_image)
    img = torch.unsqueeze(input_image, 0)
    n, c, h, w = img.size()
    ff = torch.FloatTensor(n,1024).zero_()
    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        outputs = model(input_img)
        f = outputs[2].data.cpu()
        ff = ff+f
    # norm feature
    # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)   # L2 normalize
    # ff = ff.div(fnorm.expand_as(ff))
    ff /= 2.0
    features = torch.squeeze(torch.cat((features,ff), 0))
    # print('n=%s   c=%s   h=%s   w=%s    len(features) = %s' % (n,c,h,w, len(features)))
    return features, label

# def get_one_softlabel(path, model=model):
#     input_image = Image.open(path)
#     file = os.path.split(path)[-1]
#     input_image = data_transforms['val'](input_image)
#     input_image = torch.unsqueeze(input_image, 0)
#     if use_gpu:
#         input_image = input_image.cuda()
#     outputs = model(input_image)
#     outputs = outputs[2]
#     pred_label = torch.squeeze(outputs)
#     hard_label = torch.argmax(pred_label, 0)
#     soft_label = F.softmax(pred_label, 0)
#     soft_label = soft_label.detach().cpu().numpy()
#
#
#     return soft_label, hard_label

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(751, istrain=False)
else:
    model_structure = ft_net(751)
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()
model.model2.fc = nn.Sequential()
model.classifier2 = nn.Sequential()
model.fc = nn.Sequential()
model.classifier3 = nn.Sequential()
model.fc3 = nn.Sequential()
model.classifier4 = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

src_path = 'data/market/pytorch/train_all'
files = glob.glob(src_path+'/*/*.jpg')
print(len(files))
cam_num = 6
cam_feature = np.zeros((6, 1024))
cam_cnt = np.zeros((6,))
cnt = 0
for file in files:
    feature, label = extract_one_feature(file, model)
    cam_feature[label] += feature
    cam_cnt[label] += 1
    if (cnt+1) % 1000 == 0:
        print('cnt = %d' % cnt)
        # break
    cnt += 1

for i in range(6):
    cam_feature[i] /= cam_cnt[i]
    print('cam_cnt_%d = %4d  cam_feature_%d = %s' % (i, cam_cnt[i], i, cam_feature[i]))
    print('sum = %.3f' % (np.sum(cam_feature[i])))
    print('max = %.5f  index = %4d' % (np.max(cam_feature[i]), np.argmax(cam_feature[i])))
    print('min = %.5f  index = %4d' % (np.min(cam_feature[i]), np.argmin(cam_feature[i])))

dst_path = 'data/market/pytorch'
c = np.save(os.path.join(dst_path, 'cam_features_no_norm.npy'), cam_feature)
f = np.load(os.path.join(dst_path, 'cam_features_no_norm.npy'))
print(f)



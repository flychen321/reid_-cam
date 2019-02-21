# -*- coding: utf-8 -*-
'''
this is the baseline,  if do not add gen_0000 folder(generateed images by DCGAN) under the training set,
so the LSRO equals to crossentropy loss, and the generated_image_size is 0. else the loss function will use the generated images, the loss function for
the generated images and original images are not the same.
'''
from __future__ import print_function, division
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net, ft_net_dense
from random_erasing import RandomErasing
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from cal_cam_feature import cal_camfeatures

######################################################################
# Options
parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--gpu_ids',default='3', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_DesNet121', type=str, help='output model name')
parser.add_argument('--data_dir', default='data/market/pytorch', type=str, help='training dir path')
parser.add_argument('--batchsize', default=24, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0.8, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--modelname', default='', type=str, help='save model name')

opt = parser.parse_args()

opt.use_dense = True

data_dir = opt.data_dir
name = opt.name
print('modelname in train_baseline= %s' % opt.modelname)
generated_image_size = 0
'''
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
'''
######################################################################
transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(144, interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]


transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}


def load_network(network):
    save_path = 'model/ft_DesNet121/whole_net_best_stage_2.pth'
    print('load pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


def load_network_easy(network):
    save_path = os.path.join('./model', name, 'net_best.pth')
    # save_path = 'model/model_backup/sperate/stage1_r92.82_m81.22_rer94.18_m91.23.pth'
    # save_path = 'model/model_backup/sperate/stage_2_r90.44_m79.01_rer92.46_m89.74.pth'
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)



class CamDataset(ImageFolder):
    """
    Train: For each sample creates randomly 4 images
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, transform):
        super(CamDataset, self).__init__(root, transform)
        self.labels = np.array(self.imgs)[:, 1].astype(int)
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        cams = []
        for s in self.imgs:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        img = self.data[index].item()
        img = default_loader(img)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index].item()
        cam = self.cams[index].item()
        return img, label, cam

    def get_cams_num(self):
        return np.max(self.cams) + 1

    def __len__(self):
        return len(self.imgs)


dataset_sizes = {}
# dataset_train_dir = os.path.join(data_dir, 'train_new')
# dataset_val_dir = os.path.join(data_dir, 'val_new')
dataset_train_dir = os.path.join(data_dir, 'train_all')
dataset_val_dir = os.path.join(data_dir, 'val')
dataset_sizes['train'] = sum(len(os.listdir(os.path.join(dataset_train_dir, i))) for i in os.listdir(dataset_train_dir))
dataset_sizes['val'] = sum(len(os.listdir(os.path.join(dataset_val_dir, i))) for i in os.listdir(dataset_val_dir))

print(dataset_sizes['train'])
print(dataset_sizes['val'])

dataloaders = {}
dataloaders['train'] = DataLoader(CamDataset(dataset_train_dir, data_transforms['train']), batch_size=opt.batchsize,
                                  shuffle=True, num_workers=8)
dataloaders['val'] = DataLoader(CamDataset(dataset_val_dir, data_transforms['val']), batch_size=opt.batchsize,
                                shuffle=True, num_workers=8)

id_num = len(dataloaders['train'].dataset.classes)
cam_num = int(dataloaders['train'].dataset.get_cams_num())

use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=35, stage=1, refine=False):
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    for epoch in range(num_epochs):
        print('Stage = %s' % stage)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_org_mid = 0
            running_corrects_cam = 0
            running_corrects_wo_cam = 0
            running_corrects_6cams = 0
            running_corrects_cam_mid = 0
            running_corrects_wo_mid = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, labels_cam = data

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    labels_cam = labels_cam.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'train':
                    result = model(inputs)
                else:
                    with torch.no_grad():
                        result = model(inputs)
                # result = model(inputs)
                outputs = result[0]
                outputs_cam = result[1]
                outputs_wo = result[2]
                outputs_6cams = result[3]
                outputs_org_mid = result[4]
                outputs_cam_mid = result[5]
                outputs_wo_mid = result[6]
                _, preds = torch.max(outputs.detach(), 1)  # outputs.detach()  return the index of the biggest value in each row
                _, preds_org_mid = torch.max(outputs_org_mid.detach(), 1)  # outputs.detach()  return the index of the biggest value in each row
                _, preds_cam = torch.max(outputs_cam.detach(), 1)
                _, preds_cam_mid = torch.max(outputs_cam_mid.detach(), 1)
                _, preds_wo = torch.max(outputs_wo.detach(), 1)
                _, preds_wo_mid = torch.max(outputs_wo_mid.detach(), 1)

                loss_org = criterion(outputs, labels)
                loss_cam = criterion(outputs_cam, labels_cam)
                loss_wo = criterion(outputs_wo, labels)
                loss_org_mid = criterion(outputs_org_mid, labels)
                loss_cam_mid = criterion(outputs_cam_mid, labels_cam)
                loss_wo_mid = criterion(outputs_wo_mid, labels)

                loss_6cams = torch.Tensor(outputs_6cams.shape[0])
                preds_6cams = torch.LongTensor(outputs_6cams.shape[0], labels.shape[0]).zero_().cuda()
                cam_start = 0
                cam_end = cam_num
                for i in range(cam_start, cam_end):
                    _6cams, preds_6cams[i] = torch.max(outputs_6cams[i].detach(), 1)
                    loss_6cams[i] = criterion(outputs_6cams[i], labels)
                ratio = 1.0
                if stage == 1:
                    loss = ratio * loss_org + loss_org_mid
                elif stage == 2:
                    loss = ratio * (loss_cam + loss_wo) + loss_cam_mid + loss_wo_mid
                elif stage == 3:
                    loss = torch.mean(loss_6cams)
                elif stage == 12:
                    loss = ratio * loss_org + loss_org_mid + ratio * (loss_cam + loss_wo) + loss_cam_mid + loss_wo_mid
                else:
                    print('stage = %d error!' % stage)
                    exit()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()

                running_corrects += torch.sum(preds == labels.detach())
                running_corrects_org_mid += torch.sum(preds_org_mid == labels.detach())
                running_corrects_cam += torch.sum(preds_cam == labels_cam.detach())
                running_corrects_wo_cam += torch.sum(preds_wo == labels.detach())
                running_corrects_cam_mid += torch.sum(preds_cam_mid == labels_cam.detach())
                running_corrects_wo_mid += torch.sum(preds_wo_mid == labels.detach())

                for i in range(cam_start, cam_end):
                    running_corrects_6cams += torch.sum(preds_6cams[i] == labels.detach())

            print('loss_org     = %.5f' % loss_org.detach())
            print('loss_org_mid = %.5f' % loss_org_mid.detach())
            print('loss_cam     = %.5f' % loss_cam.detach())
            print('loss_wo_cam  = %.5f' % loss_wo.detach())
            print('loss_cam_mid = %.5f' % loss_cam_mid.detach())
            print('loss_wo_mid  = %.5f' % loss_wo_mid.detach())
            print('loss_6cams[0]  = %.5f' % loss_6cams[0].detach())
            print('loss_6cams  = %s' % loss_6cams.detach())
            print('loss  = %s' % loss.detach())
            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc_org = float(running_corrects) / dataset_sizes[phase]
            epoch_acc_org_mid = float(running_corrects_org_mid) / dataset_sizes[phase]
            epoch_acc_cam = float(running_corrects_cam) / dataset_sizes[phase]
            epoch_acc_wo = float(running_corrects_wo_cam) / dataset_sizes[phase]
            epoch_acc_cam_mid = float(running_corrects_cam_mid) / dataset_sizes[phase]
            epoch_acc_wo_mid = float(running_corrects_wo_mid) / dataset_sizes[phase]
            epoch_acc_6cams = float(running_corrects_6cams / (cam_end - cam_start)) / dataset_sizes[phase]

            if stage == 1:
                r1 = 0.5
                r2 = 0.5
                epoch_acc = r1 * epoch_acc_org + r2 * epoch_acc_org_mid
            elif stage == 2:
                r1 = 0.25
                r2 = 0.25
                r3 = 0.25
                r4 = 0.25
                epoch_acc = r1 * epoch_acc_cam + r2 * epoch_acc_wo + r3 * epoch_acc_cam_mid + r4 * epoch_acc_wo_mid
            elif stage == 3:
                epoch_acc = epoch_acc_6cams
            elif stage == 12:
                r1 = 0.5
                r2 = 0.5
                epoch_acc = r1 * (epoch_acc_org + epoch_acc_org_mid)/2.0 \
                            + r2 * (epoch_acc_cam + epoch_acc_wo + epoch_acc_cam_mid + epoch_acc_wo_mid)/4.0
            else:
                print('stage = %d error!' % stage)
                exit()

            print('acc_org        =  %.5f' % epoch_acc_org)
            print('acc_org_mid    =  %.5f' % epoch_acc_org_mid)
            print('acc_cam        =  %.5f' % epoch_acc_cam)
            print('acc_wo_cam     =  %.5f' % epoch_acc_wo)
            print('acc_cam_mid    =  %.5f' % epoch_acc_cam_mid)
            print('acc_wo_mid     =  %.5f' % epoch_acc_wo_mid)
            print('acc_6cams      =  %.5f' % epoch_acc_6cams)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = model.state_dict()
                if epoch >= 0:
                    save_network(model, epoch)
            #    draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val epoch: {:d}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best')
    save_network(model, 'best' + '_stage_' + str(stage))
    save_network(model, opt.modelname)
    torch.save(model, os.path.join('./model', name, 'whole_net_best.pth'))
    torch.save(model, os.path.join('./model', name, 'whole_net_best' + '_stage_' + str(stage)) + '.pth')

    return model


# print('------------'+str(len(clas_names))+'--------------')
if True:  # opt.use_dense:
    model = ft_net_dense(id_num, cam_num, istrain=True)  # 751 class for training data in all dataset in total
    # model = ft_net_dense(751, 6, istrain=True)  # 751 class for training data in market 1501 in total
    # model = ft_net_dense(702, 8, istrain=True)  # 751 class for training data in duke in total
else:
    model = ft_net(751)

# print(model)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 40 epochs

dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

def stage_params(model=model):
    stage_1_id = list(map(id, model.model.parameters())) \
                 + list(map(id, model.org_fc.parameters())) \
                 + list(map(id, model.org_classifier.parameters())) \
                 + list(map(id, model.org_mid_fc.parameters())) \
                 + list(map(id, model.org_mid_classifier.parameters()))
    stage_1_base_id = list(map(id, model.model.parameters()))
    stage_1_base_params = filter(lambda p: id(p) in stage_1_base_id, model.parameters())
    stage_1_classifier_params = filter(lambda p: id(p) in stage_1_id and id(p) not in stage_1_base_id,
                                       model.parameters())

    stage_2_id = list(map(id, model.model2.parameters())) \
                 + list(map(id, model.cam_fc.parameters())) \
                 + list(map(id, model.cam_classifier.parameters())) \
                 + list(map(id, model.wo_rf.parameters())) \
                 + list(map(id, model.wo_fc.parameters())) \
                 + list(map(id, model.wo_classifier.parameters())) \
                 + list(map(id, model.cam_mid_fc.parameters())) \
                 + list(map(id, model.cam_mid_classifier.parameters())) \
                 + list(map(id, model.wo_mid_rf.parameters())) \
                 + list(map(id, model.wo_mid_fc.parameters())) \
                 + list(map(id, model.wo_mid_classifier.parameters()))

    stage_2_base_id = list(map(id, model.model2.parameters()))
    stage_2_base_params = filter(lambda p: id(p) in stage_2_base_id, model.parameters())
    stage_2_classifier_params = filter(lambda p: id(p) in stage_2_id and id(p) not in stage_2_base_id,
                                       model.parameters())

    stage_3_id = list(map(id, model.mask0.parameters())) \
                 + list(map(id, model.mask1.parameters())) \
                 + list(map(id, model.mask2.parameters())) \
                 + list(map(id, model.mask3.parameters())) \
                 + list(map(id, model.mask4.parameters())) \
                 + list(map(id, model.mask5.parameters())) \
                 + list(map(id, model.mul_cam_rf.parameters())) \
                 + list(map(id, model.mul_cam_fc.parameters())) \
                 + list(map(id, model.mul_cam_classifier.parameters()))
    stage_3_params = filter(lambda p: id(p) in stage_3_id, model.parameters())

    return stage_1_base_params, stage_1_classifier_params, stage_2_base_params, stage_2_classifier_params, stage_3_params


stage_1_train = True
stage_2_train = True
stage_3_train = False
stage_12_train = False

if stage_1_train:
    # model = load_network_easy(model)
    stage_1_base_params, stage_1_classifier_params, stage_2_base_params, stage_2_classifier_params, \
    stage_3_params = stage_params(model)
    epoc = 130
    lr_ratio = 1
    step = 40
    optimizer_ft = optim.SGD([
        {'params': stage_1_base_params, 'lr': 0.01 * lr_ratio},
        {'params': stage_1_classifier_params, 'lr': 0.05 * lr_ratio},
        {'params': stage_2_base_params, 'lr': 0},
        {'params': stage_2_classifier_params, 'lr': 0},
        {'params': stage_3_params, 'lr': 0},
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=epoc, stage=1)


if stage_2_train:
    model = load_network_easy(model)
    stage_1_base_params, stage_1_classifier_params, stage_2_base_params, stage_2_classifier_params,\
    stage_3_params = stage_params(model)
    epoc = 130
    lr_ratio = 1
    step = 40
    optimizer_ft = optim.SGD([
        {'params': stage_1_base_params, 'lr': 0},
        {'params': stage_1_classifier_params, 'lr': 0},
        {'params': stage_2_base_params, 'lr': 0.01 * lr_ratio},
        {'params': stage_2_classifier_params, 'lr': 0.05 * lr_ratio},
        {'params': stage_3_params, 'lr': 0},
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=epoc, stage=2)


if stage_12_train:
    # model = load_network_easy(model)
    stage_1_base_params, stage_1_classifier_params, stage_2_base_params, stage_2_classifier_params, \
    stage_3_params = stage_params(model)
    epoc = 130
    lr_ratio = 1
    step = 40
    optimizer_ft = optim.SGD([
        {'params': stage_1_base_params, 'lr': 0.01 * lr_ratio},
        {'params': stage_1_classifier_params, 'lr': 0.05 * lr_ratio},
        {'params': stage_2_base_params, 'lr': 0.01 * lr_ratio},
        {'params': stage_2_classifier_params, 'lr': 0.05 * lr_ratio},
        {'params': stage_3_params, 'lr': 0},
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=epoc, stage=12)


if stage_3_train:
    model = load_network(model)
    stage_1_base_params, stage_1_classifier_params, stage_2_base_params, stage_2_classifier_params, \
    stage_3_params = stage_params(model)
    # cal_camfeatures()
    epoc = 20
    lr_ratio = 1
    step = 6
    optimizer_ft = optim.SGD([
        {'params': stage_1_base_params, 'lr': 0},
        {'params': stage_1_classifier_params, 'lr': 0},
        {'params': stage_2_base_params, 'lr': 0},
        {'params': stage_2_classifier_params, 'lr': 0},
        {'params': stage_3_params, 'lr': 0.05 * lr_ratio},
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=epoc, stage=3)


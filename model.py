import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from scipy.io import loadmat
import os
import numpy as np

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net,self).__init__()
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        num_ftrs = model_ft.fc.in_features  # extract feature parameters of fully collected layers
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs, num_bottleneck)]   # add a linear layer, batchnorm layer, leakyrelu layer and dropout layer
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  #default dropout rate 0.5
        #transforms.CenterCrop(224),
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]  # class_num classification
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ReFineBlock(nn.Module):
    def __init__(self, input_dim=1024, dropout=True, relu=True, num_bottleneck=1024, layer=2):
        super(ReFineBlock, self).__init__()
        add_block = []
        for i in range(layer):
            add_block += [nn.Linear(input_dim, num_bottleneck)]
            add_block += [nn.BatchNorm1d(num_bottleneck)]
            if relu:
                add_block += [nn.LeakyReLU(0.1)]
            if dropout:
                add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x

class FcBlock(nn.Module):
    def __init__(self, input_dim=1024, dropout=True, relu=True, num_bottleneck=512):
        super(FcBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751):
        super(ClassBlock, self).__init__()
        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x


class ft_net_dense(nn.Module):
    def __init__(self, class_num, cam_num=6, istrain=True, ratio=0.65):
        super(ft_net_dense, self).__init__()
        dst_path = 'data/market/pytorch'
        c = np.load(os.path.join(dst_path, 'cam_features_no_norm.npy'))
        self.cam_f_info = torch.from_numpy(c).cuda()
        self.class_num = class_num
        self.cam_num = cam_num
        self.istrain = istrain
        self.ratio = ratio
        model_ft = models.densenet121(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = FcBlock()
        self.model = model_ft
        self.classifier = ClassBlock()
        model_ft2 = models.densenet121(pretrained=True)
        model_ft2.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft2.fc = FcBlock()
        self.model2 = model_ft2
        self.classifier2 = ClassBlock(class_num=self.cam_num)
        self.fc = FcBlock()
        self.classifier3 = ClassBlock()
        self.rf = ReFineBlock(layer=2)
        self.fc3 = FcBlock()
        self.classifier4 = ClassBlock()



    def forward(self, x):
        temp = x
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        # x = x.div(torch.norm(x, p=2, dim=1, keepdim=True).expand_as(x))
        feature_1 = x
        x = self.model.fc(x)
        x = self.classifier(x)

        y = self.model2.features(temp)
        y = y.view(y.size(0), -1)
        # y = y.div(torch.norm(y, p=2, dim=1, keepdim=True).expand_as(y))
        feature_2 = y
        y = self.model2.fc(y)
        y = self.classifier2(y)

        z_mid = feature_1 - feature_2
        # z = z.div(torch.norm(z, p=2, dim=1, keepdim=True).expand_as(z))
        z = self.fc(z_mid)
        z = self.classifier3(z)

        # for i in range(self.cam_num):   # for train 6cam
        #     temp = z_mid + ratio * self.cam_f_info[i]
        #     temp = self.rf(temp)
        #     temp = self.fc3(temp)
        #     temp = self.classifier4(temp)
        #     if i == 0:
        #         result = temp.unsqueeze(0)
        #     else:
        #         result = torch.cat((result, temp.unsqueeze(0)), 0)

        # mid = feature_1 - 0.35*feature_2
        # temp = feature_1 - 0.35*feature_2  # for train  7cam
        # temp = self.rf(temp)
        # temp = self.fc3(temp)
        # temp = self.classifier4(temp)
        # result = temp.unsqueeze(0)
        # for i in range(self.cam_num):
        #     temp = mid + 0.35*self.cam_f_info[i]
        #     temp = self.rf(temp)
        #     temp = self.fc3(temp)
        #     temp = self.classifier4(temp)
        #     result = torch.cat((result, temp.unsqueeze(0)), 0)

        mid = feature_1 - self.ratio * feature_2   # for test  7cam
        for i in range(self.cam_num):
            temp = mid + self.ratio * self.cam_f_info[i].float()
            temp = self.rf(temp)
            temp = self.fc3(temp)
            temp = self.classifier4(temp)
            if i == 0:
                result = temp.unsqueeze(0)
            else:
                result = torch.cat((result, temp.unsqueeze(0)), 0)
        if not self.istrain:
            result = result.transpose(0, 1)
            result = result.contiguous().view(result.size(0), -1)



        return x, y, z, result
'''
input = Variable(torch.FloatTensor(8, 3, 224, 224))
output = net(input)
print('net output size:')
#print(output.shape)
'''
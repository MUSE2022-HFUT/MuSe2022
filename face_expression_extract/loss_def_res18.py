import sys
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torch.nn as nn
import torch.nn.init as init
from resnet import ResNet, BasicBlock, Bottleneck
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MVFace(torch.nn.Module):
    def __init__(self, backbone, num_class_1, num_class_2_list):
        super(MVFace, self).__init__()

        self.num_class_1 = num_class_1
        
        self.num_class_2_list = num_class_2_list #[3,2,3,8,8]
        self.num_class_2 = num_class_2_list[4]

        if backbone == 'resnet18':
            self.feat_net = ResNet(last_stride=2,
                            block=BasicBlock, frozen_stages=-1,
                            layers=[2, 2, 2, 2])
            self.in_planes = 512

        elif backbone == 'resnet50_ibn':
            self.feat_net = resnet50_ibn_a(last_stride=2)
            self.in_planes = 2048

        elif backbone == 'resnet18_convnext':
            self.feat_net = ResNet_ConvNext(last_stride=2,
                            block=ResConvBasicBlock, frozen_stages=-1,
                            layers=[2, 2, 6, 2])
            self.in_planes = 768
        else:
            raise Exception('backbone must be resnet18, resnet50 and resnet50_ibn.')
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc_1 = nn.Linear(self.in_planes, num_class_2_list[4])
        self.bn_1 = nn.BatchNorm1d( num_class_2_list[4] )


    def forward(self, data):
        feat = self.feat_net.forward(data)
        feat = self.gap(feat).squeeze().squeeze()

        # out1 = self.fc_1(feat)

        return feat

    def load_param(self, pretrained):
        param_dict = torch.load(pretrained, map_location=lambda storage,loc: storage.cpu())
        # print('Pretrained choice ', cfg.pretrained_choice)

        for i in param_dict['state_dict']:
            if ('loss_layer' in i) or ('fc' in i):
                continue
            else:
                self.state_dict()[i].copy_(param_dict['state_dict'][i])
  
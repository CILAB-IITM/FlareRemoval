import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

##############################################################################
# Losses
##############################################################################
class GanLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GanLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target = self.real_label
        else:
            target = self.fake_label
        targets = torch.full_like(input, fill_value=target)
        return targets

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


from torchvision.models import vgg19

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        model=model.cuda()
        model = model.eval()
        # Freeze VGG19 #
        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        
        self.criterion = nn.L1Loss()
        self.layer_weight=[1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]        
        self.selected_feature_index=[2,7,12,21,30]

    def extract_feature(self,x):
        selected_features = []
        for i,model in enumerate(self.vgg):
            x = model(x)
            if i in self.selected_feature_index:
                selected_features.append(x.clone())
        return selected_features

    def forward(self, x, y):              
        x_vgg, y_vgg = self.extract_feature(x), self.extract_feature(y)
        loss = 0
        len_feature=len(x_vgg)
        for i in range(len_feature):
            loss+=self.criterion(x_vgg[i],y_vgg[i])*self.layer_weight[i]
        return loss 
    
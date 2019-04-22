import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math


class net(nn.Module):
    ####
    # Define our own model
    ####
    def __init__(self, n_output=43):
        super(net, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, n_output)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.sigmoid(features)
        return features

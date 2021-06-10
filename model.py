#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
import torchvision.models as models
import torchvision
import os 
import torch.nn.functional as F

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class BackBone(models.AlexNet):
    def __init__(self, model_path='./pretrained/alexnet-owt-4df8aa71.pth', freeze=True, layer=2):
        super().__init__()
        if os.path.exists(model_path):
            print('Loading local checkpoint')
            self.load_state_dict(torch.load(model_path))
        else:
            print('Downloading model')
            state_dict = torchvision.utils.load_state_dict_from_url(model_urls['alexnet'], progress=True)
            self.load_state_dict(state_dict)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        if layer > 0:
            self.classifier.__delitem__(6)
        if layer > 1:
            self.classifier.__delitem__(5)
            self.classifier.__delitem__(4)
            self.classifier.__delitem__(3)
        if layer > 2:
            self.classifier = None

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

class FineTuner(nn.Module):
    def __init__(self, num_classes=50, layer_bb=2):
        super().__init__()
        self.backbone = BackBone(freeze=True, layer=layer_bb)
        self.clf = nn.Linear(4096, num_classes)
    def forward(self, x):
        return self.clf(self.backbone(x))


class ModelRegression(nn.Module):
    def __init__(self, in_dim=4097):
        super().__init__()
        slope = 0.01 # hyperpara
        hidden_dim = 512
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, in_dim)
        )
    
    def forward(self, x):
        out = self.model(x)
        return out


class ProtoNetwork(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=4096, layer_bb=2):
        super(ProtoNetwork, self).__init__()
        self.bb = BackBone(freeze=True, layer=layer_bb)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        return self.encoder(self.bb(x))
        
    def backbone_forward(self, x):
        return self.bb(x)




if __name__ == '__main__':
    bb = BackBone()
    x = torch.ones((1,3, 256, 256))
    print(bb(x).shape)
#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import os 

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class BackBone(models.AlexNet):
    def __init__(self, model_path='./pretrained/alexnet-owt-4df8aa71.pth', freeze=True):
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
        self.classifier.__delitem__(6)

class FineTuner(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.backbone = BackBone(freeze=False)
        self.clf = nn.Linear(4096, num_classes)
    def forward(self, x):
        return self.clf(self.backbone(x))


class ModelRegression(nn.Module):
    def __init__(self, in_dim=4097):
        super().__init__()
        slope = 0.01 # hyperpara
        self.model = nn.Sequential(
            nn.Linear(in_dim, 6144),
            nn.LeakyReLU(slope),
            nn.Linear(6144, 5120),
            nn.LeakyReLU(slope),
            nn.Linear(5120, 4097),
            nn.LeakyReLU(slope),
            nn.Linear(4097, in_dim)
        )
    
    def forward(self, x):
        out = self.model(x)
        return out


class ProtoNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # TODO prototype network
        return x

if __name__ == '__main__':
    # bb = BackBone()
    # x = torch.ones((1,3, 256, 256))
    # print(bb(x).shape)
    m = ModelRegression()
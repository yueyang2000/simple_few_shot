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
    def __init__(self, input_dim=4096):
        super().__init__()
        self.backbone = BackBone()
        self.transform = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Dropout(inplace=True),
            nn.Linear(4096, 4096)
        )

    def forward(self, x):
        # TODO add model regression
        F = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
        )
        return F(x)


class ProtoNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # TODO prototype network
        return x

if __name__ == '__main__':
    bb = BackBone()
    x = torch.ones((1,3, 256, 256))
    print(bb(x).shape)
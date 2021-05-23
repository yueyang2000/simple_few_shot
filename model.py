#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import os 
import torch.nn.functional as F
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
        return x 


def conv_block(in_channels, out_channels):
    '''
    return a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def euclidean_dist(x, y):
    # x:n×d
    # y:m×d
    n = x.size(0)
    d = x.size(1)
    m = y.size(0)
    assert y.size(1) == d
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return 

class ProtoNetwork(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNetwork, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim,hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
    def forward(self, x):
        # TODO prototype network
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def loss(self, y):
        n_class = len(torch.unique(y))
        n_support = 1
        n_query = 1

        target_inds = torch.arange(0, n_class)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()
        # support_indices
        # query_indices
        # prototypes
        # query_samples
        dists = euclidean_dist(query_samples, prototypes)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        # acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        return loss_val


if __name__ == '__main__':
    bb = BackBone()
    x = torch.ones((1,3, 256, 256))
    print(bb(x).shape)
from torch.optim import optimizer
from tqdm.std import tqdm
from dataset import Caltech256
import torch 
import torch.nn as nn 
import torch.optim as optim 

import argparse 

from model import ProtoNetwork


if __name__ == '__main__':

    model = ProtoNetwork() 
    trainset = Caltech256(split='train')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    
    #train
    epochs = 10
    episode_per_epoch = 10
    n_class = 30
    n_support = 3
    train_loss = []
    for epoch in range(epochs):
        print('===epoch:{}==='.format(epoch))
        model.train()
        for epi in range(episode_per_epoch):
            support, query = trainset.sample_proto_batch(n_class, n_support)
            support_embed, query_embed = model(support), model(query)
            # TODO: compute proto loss
            break
        break




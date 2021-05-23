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
    train_data = Caltech256(split='train')
    # a sampler is missing
    dataloader = torch.utils.data.Dataloader(train_data)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    
    #train
    epochs = 10
    train_loss = []
    for epoch in range(epochs):
        print('===epoch:{}==='.format(epoch))
        tr_iter = iter(dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optimizer.zero_grad()
            x, y = batch
            model_output = model(x)
            loss = model.loss(model_output, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        avg_loss = np.mean(train_loss)





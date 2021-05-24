from numpy.core.numeric import full
from torch.optim import optimizer
from tqdm import tqdm
from dataset import Caltech256
import torch 
import torch.nn as nn 
import torch.optim as optim 

from model import ProtoNetwork
from proto import proto_loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ProtoNetwork()
    trainset = Caltech256(split='train')
    testset = Caltech256(split='test')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    
    #train
    epochs = 10
    episode_per_epoch = 10
    full_class = 50
    n_class = 10
    n_support = 3
    train_loss = []
    
    full_support = trainset.full_proto_batch()
    full_query = testset.full_proto_batch()
    for epoch in range(epochs):
        print('===epoch:{}==='.format(epoch))
        model.train()
        for epi in tqdm(range(episode_per_epoch)):
            optimizer.zero_grad()
            support, query = trainset.sample_proto_batch(n_class, n_support)
            support_embed, query_embed = model(support), model(query)

            loss, acc = proto_loss(support_embed, query_embed, n_class)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full_support_embed, full_query_embed = model(full_support), model(full_query)
            loss, acc = proto_loss(full_support_embed, full_query_embed, full_class)
            print(f'Epoch {epoch}, Acc =', acc.item())

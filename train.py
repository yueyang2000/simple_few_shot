from torch.optim import optimizer
from tqdm import tqdm
from dataset import Caltech256
import torch 
import os 

from model import ProtoNetwork
from proto import proto_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ProtoNetwork().to(device)
    trainset = Caltech256(split='train')
    testset = Caltech256(split='test')

    
    #train
    epochs = 50
    episode_per_epoch = 10
    full_class = 50
    n_class = 30
    n_support = 3
    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs), 0)
    full_support = trainset.full_proto_batch().to(device)
    full_query = testset.full_proto_batch().to(device)
    best_acc = 0
    for epoch in range(epochs):
        print('===epoch:{}==='.format(epoch))
        model.train()
        train_acc = 0
        train_loss = 0
        for epi in tqdm(range(episode_per_epoch)):
            optimizer.zero_grad()
            support, query = trainset.sample_proto_batch(n_class, n_support)
            support, query = support.to(device), query.to(device)
            support_embed, query_embed = model(support), model(query)

            loss, acc = proto_loss(support_embed, query_embed, n_class, device)
            train_acc += acc
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Train\tAcc =', round(train_acc/episode_per_epoch, 4))
        print('Train\tLoss =', round(train_loss/episode_per_epoch, 4))
        model.eval()
        with torch.no_grad():
            full_support_embed, full_query_embed = model(full_support), model(full_query)
            loss, acc = proto_loss(full_support_embed, full_query_embed, full_class, device)
            print('Test\tAcc =', round(acc, 4))
            print('Test\tLoss =', round(loss.item(), 4))
            if acc > best_acc:
                best_acc = acc 
    print('Best Acc =', round(best_acc, 4))
    # Best Acc = 0.696
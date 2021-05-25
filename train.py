from torch.optim import optimizer
from torchvision.transforms.transforms import RandomRotation
from tqdm import tqdm
from dataset import Caltech256, ProtoBatchSampler
import torch
import torchvision.transforms as transforms 
import os, argparse

from model import ProtoNetwork
from proto import proto_loss




parser = argparse.ArgumentParser('Prototypical Network Training')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--aug', action='store_true', default=False)
parser.add_argument('--n_class', type=int, default=30)
parser.add_argument('--n_shot', type=int, default=10)
parser.add_argument('--n_support', type=int, default=7)
parser.add_argument('--embed_dim', type=int, default=4096)
parser.add_argument('--n_episode', type=int, default=20)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    full_class = 50
    epochs, lr, n_class, n_shot, n_support, embed_dim, n_episode = args.epochs, args.lr, args.n_class, args.n_shot, args.n_support, args.embed_dim, args.n_episode

    n_query = n_shot - n_support
    model = ProtoNetwork(embed_dim=embed_dim).to(device)
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if args.aug:
        trainset = Caltech256(split='train', transform=train_transform)
    else:
        trainset = Caltech256(split='train')
    sampler = ProtoBatchSampler(n_sample=n_shot, n_class=n_class, n_support=n_support, n_episode=n_episode)
    trainloader =  torch.utils.data.DataLoader(trainset, batch_sampler=sampler, num_workers=4)
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    else:
        print('Unsupport Optim type!')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs), 0)
    full_support = Caltech256(split='train').full_proto_batch().to(device)
    full_query = Caltech256(split='test').full_proto_batch().to(device)
    best_acc = 0
    for epoch in range(epochs):
        print('===epoch:{}==='.format(epoch))
        model.train()
        train_acc = 0
        train_loss = 0
        for batch_data, batch_label in tqdm(trainloader):
            optimizer.zero_grad()
            support, query = batch_data[:n_class*n_support], batch_data[n_class*n_support:]
            support, query = support.to(device), query.to(device)
            support_embed, query_embed = model(support), model(query)
            loss, acc = proto_loss(support_embed, query_embed, n_class, device)
            train_acc += acc
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('Train\tAcc =', round(train_acc/n_episode, 4))
        print('Train\tLoss =', round(train_loss/n_episode, 4))
        model.eval()
        with torch.no_grad():
            full_support_embed, full_query_embed = model(full_support), model(full_query)
            loss, acc = proto_loss(full_support_embed, full_query_embed, full_class, device)
            print('Test\tAcc =', round(acc, 4))
            print('Test\tLoss =', round(loss.item(), 4))
            if acc > best_acc:
                best_acc = acc 
    print('Best Acc =', round(best_acc, 4))
    # Best Acc = 0.72 with n_class=30
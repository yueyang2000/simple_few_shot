from random import shuffle
from torch.optim import optimizer
from tqdm import tqdm
from dataset import Caltech256, Caltech256Aug, ProtoBatchSampler, Proj2Test
import torch 
import os, argparse, json 

from model import ProtoNetwork, BackBone
from loss import proto_loss, nca_loss, centriod_pred, knn_pred, soft_assign_pred


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_best_acc():
    with open('best_acc.txt', 'r') as f:
        return float(f.read())

def update_best_acc(acc):
    with open('best_acc.txt', 'w') as f:
        f.write(str(round(acc, 4)))

proj2_testloader = torch.utils.data.DataLoader(Proj2Test(), shuffle=False, batch_size=50)

def generate_result(model, support):
    with open('proj2_prediction_14.txt', 'w') as f:
        model.eval()
        with torch.no_grad():
            all_pred = []
            for batch_data in proj2_testloader:
                batch_data = batch_data.to(device)
                query_embed = model(batch_data)
                pred = centriod_pred(support, query_embed, 50)
                all_pred.append(pred)
            # label 1~50
            all_pred = (torch.cat(all_pred)+1).cpu().numpy().tolist()
            for p in all_pred:
                f.write(str(int(p)) + '\n')



parser = argparse.ArgumentParser('Prototypical Network Training')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_class', type=int, default=50)
parser.add_argument('--n_shot', type=int, default=10)
parser.add_argument('--n_support', type=int, default=7)
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--n_episode', type=int, default=20)
parser.add_argument('--loss_type', type=str, default='proto', help='proto/nca')
parser.add_argument('--dist_type', type=str, default='cosine', help='cosine/euclidean')
parser.add_argument('--test_bb', action='store_true', default=False)
parser.add_argument('--layer_bb', type=int, default=2)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
   
    full_class = 50
    epochs, lr, n_class, n_shot, n_support, embed_dim, n_episode = args.epochs, args.lr, args.n_class, args.n_shot, args.n_support, args.embed_dim, args.n_episode

    n_query = n_shot - n_support
    
    n_aug = 4
    n_sample = n_shot * n_aug 
    n_support = n_support * n_aug

    trainset = Caltech256Aug(split='train')
    testset = Caltech256(split='test')
    sampler = ProtoBatchSampler(n_sample=n_sample, n_class=n_class, n_support=n_support, n_episode=n_episode)
    trainloader =  torch.utils.data.DataLoader(trainset, batch_sampler=sampler, num_workers=4)
    

    full_support = trainset.full_proto_batch()
    full_query = testset.full_proto_batch()
    full_query_labels = testset.get_labels().to(device)
    best_acc = 0

    if args.test_bb:
        model = BackBone(freeze=True, layer=args.layer_bb).to(device)
        model.eval()
        with torch.no_grad():
            full_support_embed = []
            for i in range(full_class):
                support = full_support[i*n_sample:(i+1)*n_sample]
                support = support.to(device)
                support_embed = model(support)
                full_support_embed.append(support_embed)
            full_support_embed = torch.cat(full_support_embed)
            acc_centriod, acc_knn, acc_soft = 0., 0., 0.
            for i in range(30):
                query = full_query[i::30]
                query = query.to(device)
                query_labels = full_query_labels[i::30]
                query_embed = model(query)
                pred = centriod_pred(full_support_embed, query_embed, full_class, args.dist_type)
                acc_centriod += torch.eq(pred, query_labels).float().mean().item()
                pred = knn_pred(20, full_support_embed, query_embed, full_class, args.dist_type)
                acc_knn += torch.eq(pred, query_labels).float().mean().item()
                pred = soft_assign_pred(full_support_embed, query_embed, full_class, args.dist_type)
                acc_soft += torch.eq(pred, query_labels).float().mean().item()
                
            acc_centriod = round(acc_centriod / 30, 4)
            acc_knn = round(acc_knn / 30, 4)
            acc_soft = round(acc_soft / 30, 4)
            print('acc_centriod:', acc_centriod)
            print('acc_knn:', acc_knn)
            print('acc_soft:', acc_soft)
    else:
        if embed_dim == 0:
            model = BackBone(freeze=False, layer=args.layer_bb).to(device)
        else:
            model = ProtoNetwork(embed_dim=embed_dim, layer_bb=args.layer_bb).to(device)
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        else:
            print('Unsupport Optim type!')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs), 0)
        for epoch in range(epochs):
            print('===epoch:{}==='.format(epoch))
            model.train()
            train_acc = 0
            train_loss = 0
            for batch_data, batch_label in tqdm(trainloader):
                optimizer.zero_grad()
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
                support, query = batch_data[:n_class*n_support], batch_data[n_class*n_support:]
                #support, query = support.to(device), query.to(device)
                support_embed, query_embed = model(support), model(query)
                if args.loss_type == 'proto':
                    loss, acc = proto_loss(support_embed, query_embed, n_class)
                    train_acc += acc
                elif args.loss_type == 'nca':
                    loss = nca_loss(support_embed, batch_label[:n_class*n_support])
                else:
                    print('Unsupport loss type!')
                    break
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()
            if args.loss_type == 'proto': print('Train\tAcc =', round(train_acc/n_episode, 4))
            print('Train\tLoss =', round(train_loss/n_episode, 4))
            model.eval()
            with torch.no_grad():
                full_support_embed = []
                for i in range(full_class):
                    support = full_support[i*n_sample:(i+1)*n_sample]
                    support = support.to(device)
                    support_embed = model(support)
                    full_support_embed.append(support_embed)
                full_support_embed = torch.cat(full_support_embed)
                all_acc = 0
                for i in range(30):
                    query = full_query[i::30]
                    query = query.to(device)
                    query_labels = full_query_labels[i::30]
                    query_embed = model(query)
                    pred = centriod_pred(full_support_embed, query_embed, full_class, args.dist_type)
                    acc = torch.eq(pred, query_labels).float().mean().item()
                    all_acc += acc
                acc = round(all_acc / 30, 4)
                print('Test\tAcc =', acc)
                if acc > best_acc:
                    best_acc = acc 
                if get_best_acc() < best_acc:
                    print('New high score!!!')
                    generate_result(model, full_support_embed)
                    update_best_acc(acc)
                    with open('args.txt', 'w') as f:
                        json.dump(args.__dict__, f, indent=2)
        print('Best Acc =', round(best_acc, 4))

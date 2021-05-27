import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from model import ModelRegression
<<<<<<< HEAD
from dataset import BaseFeature
=======
from dataset import ModelPair, BaseFeature

bf = BaseFeature()
>>>>>>> 5e5bd557f1613e2267f2aa65f160e04e917337f3

class mrnLoss(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam
        self.reg = nn.MSELoss(reduction='sum')
<<<<<<< HEAD
    
    def forward(self, w0, w_star, c):
        reg_loss = self.reg(w0, w_star)
        # TODO: performance loss
        for i in range(w0.shape[0]):
            # pred = 
            c_idx = c[i]
            pass
        return reg_loss
=======

    def forward(self, w_preds, w_stars, labels):
        reg_loss = self.reg(w_preds, w_stars)
        perf_loss = 0
        for c_id in range(len(labels)):
            label = int(labels[c_id])
            w_pred = w_preds[c_id]
            x, y = bf[label-1]
            for idx in range(len(x)):
                perf_loss += 1 - (y[idx]*torch.matmul(w_pred.t(), torch.from_numpy(x[idx]).to(torch.float32)))
        return reg_loss + self.lam * perf_loss
>>>>>>> 5e5bd557f1613e2267f2aa65f160e04e917337f3


def train():
    epochs = 50

    model = ModelRegression()
    mrn_loss = mrnLoss()

    train_data = ModelPair()
    trainloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    # test_data = Caltech256(split='test')
    # testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    # best_acc = 0
    for epoch in range(epochs):
        print('===epoch:{}==='.format(epoch))
        model.train()
        train_acc = 0
        train_loss = 0
        for w0, w_star, class_index in tqdm(trainloader):
            w0 = w0.to(torch.float32)
            w_star = w_star.to(torch.float32)
            pred = model(w0)
            loss = mrn_loss(pred, w_star, class_index)
            train_loss += loss
            train_acc += np.sum(np.square(pred.detach().numpy(), w_star.detach().numpy()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss: {:.6f} - \tAcc: {:.6f}'.format(
            train_loss/len(trainloader), train_acc/len(trainloader)))
        # 保存检查点
        torch.save(model, 'models/epoch{}_checkpoint.pkl'.format(epoch))


if __name__ == '__main__':
    train()
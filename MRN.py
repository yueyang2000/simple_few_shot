import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from model import ModelRegression
from dataset import ModelPair, BaseFeatureDataset, Caltech256

from baseline import get_all_features

bf = BaseFeatureDataset()

class mrnLoss(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam
        self.reg = nn.MSELoss(reduction='sum')

    def forward(self, w_preds, w_stars, labels):
        reg_loss = self.reg(w_preds, w_stars)
        perf_loss = 0
        for c_id in range(len(labels)):
            label = int(labels[c_id])
            w_pred = w_preds[c_id]
            x, y = bf[label-1]
            perf_loss += (1 - torch.sum(x * w_pred, dim=1) * y).sum()
        return reg_loss + self.lam * perf_loss


def train():
    epochs = 50

    model = ModelRegression()
    mrn_loss = mrnLoss()

    train_data = ModelPair()
    trainloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    test_w0 = np.load("data/Caltech256_w0.npy")
    x_test, y_test = get_all_features(split='test')
    x_test = np.append(x_test, np.ones((x_test.shape[0], 1)), axis=1)  # (1500, 4097)

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
            train_acc += np.sum(np.square(pred.detach().numpy() - w_star.detach().numpy()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Loss: {:.6f} - \tAcc: {:.6f}'.format(
            train_loss/len(trainloader), train_acc/len(trainloader)))

        pred = np.argmax(np.matmul(x_test, model(test_w0).T), axis=1)
        total = x_test.shape[0]
        correct = (pred == y_test).sum().item()
        test_acc = correct / total
        print("test Acc: {:.6f}".format(test_acc))
        # 保存检查点
        torch.save(model, 'models/epoch{}_checkpoint.pkl'.format(epoch))


def test(w0):
    # w0 [50, 4097] ndarray, sort by class index
    # class_idx [batch_size], 1 <= item in class_idx <= 1000
    x_test, y_test = get_all_features(split='test')
    x_test = np.append(x_test, np.ones((x_test.shape[0], 1)), axis=1) # (1500, 4097)
    pred = np.argmax(np.matmul(x_test, w0.T), axis=1)
    total = x_test.shape[0]
    correct = (pred == y_test).sum().item()
    acc = correct / total
    return acc

def refine(wT,):
    epochs = 64
    # TODO


if __name__ == '__main__':
    train()
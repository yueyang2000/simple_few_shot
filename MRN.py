import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModelRegression
from dataset import BaseFeature

class mrnLoss(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam
        self.reg = nn.MSELoss(reduction='sum')
    
    def forward(self, w0, w_star, c):
        reg_loss = self.reg(w0, w_star)
        # TODO: performance loss
        for i in range(w0.shape[0]):
            # pred = 
            c_idx = c[i]
            pass
        return reg_loss


def train():
    pass


if __name__ == '__main__':
    pass
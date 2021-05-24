import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModelRegression

class mrnLoss(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam
        self.reg = nn.MSELoss(reduction='sum')
    
    def forward(self, w0, w_star):
        reg_loss = self.reg(w0, w_star)
        # TODO: performance loss
        return reg_loss


if __name__ == '__main__':
    pass
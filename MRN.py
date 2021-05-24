import torch
import torch.nn as nn
import torch.nn.functional as F

class mrnLoss(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam
        self.reg_loss = nn.MSELoss(reduction='sum')
    
    def forward(self, x):
        return x


if __name__ == '__main__':
    pass
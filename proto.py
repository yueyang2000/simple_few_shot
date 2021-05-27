import torch
from torch.nn import functional as F


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def proto_loss(support, query, n_class, device='cpu'):
    n_support = support.shape[0] // n_class
    n_query = query.shape[0] // n_class
    fdim = support.shape[1]
    support = support.view(n_class, n_support, fdim)
    prototypes =support.mean(dim=1) # [n_class, fdim]

    target_inds = torch.arange(0, n_class).to(device)
    target_inds = target_inds.view(n_class, 1, 1)
    target_inds = target_inds.expand(n_class, n_query, 1).long() #[n_class, n_query, 1]

    dists = cosine_dist(query, prototypes) #[n_query, n_class]

    log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1) #[n_class, n_query, n_class]
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2) #[n_class, n_query]
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean().item()
    return loss_val, acc_val

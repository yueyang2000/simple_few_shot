import torch
from torch._C import device
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

def proto_loss(support, query, n_class):
    device = support.device

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

DIST = {
    'cosine': cosine_dist,
    'euclidean': euclidean_dist
}

def nca_loss(data, targets):
    device = data.device

    B = data.shape[0]
    dists = cosine_dist(data, data) #[B,B]
    neg_exp_dists = torch.exp(-dists)
    losses = []
    all_idx = torch.arange(0, B).to(device)
    for b in range(B):
        scores = neg_exp_dists[b]
        # should have at least 2 sample per class
        same = torch.logical_and(targets == targets[b], all_idx != b)
        b_loss = -torch.log(scores[same].sum() / scores[all_idx!=b].sum())
        losses.append(b_loss)
    return torch.mean(torch.stack(losses))


def centriod_pred(support, query, n_class, dist_type='cosine'):
    n_support = support.shape[0] // n_class
    fdim = support.shape[1]
    support = support.view(n_class, n_support, fdim)
    prototypes = support.mean(dim=1) # [n_class, fdim]
    dists = DIST[dist_type](query, prototypes)
    _, pred = dists.min(1)
    return pred

def knn_pred(k, support, query, n_class, dist_type='cosine'):
    device = support.device
    dists = DIST[dist_type](query, support)
    n_support = support.shape[0] // n_class
    support_inds = torch.arange(0, n_class).to(device)
    support_inds = support_inds.view(n_class, 1)
    support_inds = support_inds.expand(n_class, n_support).long()
    support_inds = support_inds.reshape(n_class * n_support)
    pred = []
    for q in range(query.shape[0]):
        q_dists =  dists[q]
        _, indices = torch.topk(q_dists, largest=False, k=k)
        cls_inds = support_inds[indices]
        cnt = torch.bincount(cls_inds)
        _, idx = torch.max(cnt, dim=0)
        pred.append(idx.item())
    pred = torch.tensor(pred, device=device).long()
    return pred

def soft_assign_pred(support, query, n_class, dist_type='cosine'):
    device = support.device
    n_support = support.shape[0] // n_class
    dists = DIST[dist_type](query, support)
    dists = dists.view(query.shape[0], n_class, n_support)
    exp_neg_dists = torch.exp(-dists)
    logits = exp_neg_dists.sum(dim=2)
    _, pred = logits.max(dim=1)
    return pred
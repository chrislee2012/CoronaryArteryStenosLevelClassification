import torch.nn as nn
import torch.nn.functional as F


class OHEMLoss(nn.Module):
    def __init__(self, k=1):
        super(OHEMLoss, self).__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        loss = F.cross_entropy(y_pred, y_true, reduction='none')
        if len(y_pred) < self.k:
            k = len(y_pred)
            _, idxs = loss.topk(k)
        else:
            _, idxs = loss.topk(self.k)
        return loss[idxs].mean()

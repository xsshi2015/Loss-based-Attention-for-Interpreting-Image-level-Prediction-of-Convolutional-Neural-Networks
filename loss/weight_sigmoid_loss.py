import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sigmoid_with_weights(outputs, target, weights=None):
    assert outputs.dim() == 2
    assert not target.requires_grad
    assert target.dim() == 2
    # import pdb; pdb.set_trace()

    outputs = F.sigmoid(outputs)
    outputs = torch.clamp(outputs, min=1e-6, max=1-1e-6)

    loss = target*torch.log(outputs)

    s_power = target.sum(1)

    loss_1 = torch.squeeze(Variable(torch.zeros(outputs.size(0),1).cuda().float()))

    for iter in range(1, max(s_power.long())+1):
        idx = torch.squeeze(torch.nonzero(s_power==float(iter)))
        if iter==1:
            loss_1[idx] = -1.0*loss[idx,:].sum(1)
        else:
            temp_idx = (torch.squeeze(torch.nonzero(target[idx,:]))[:,1]).view(idx.size(0), iter)
            temp_loss = loss[idx, temp_idx[:,0]]
            loss_1[idx] = loss[idx, temp_idx[:,0]]
            for jter in range(1,iter):
                loss_1[idx] = torch.max(loss_1[idx], torch.squeeze(loss[idx, temp_idx[:,jter]]))

            loss_1[idx] = loss_1[idx]*torch.sign(loss_1[idx])

    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss_1.size()) == list(weights.size())
        # Weight the loss
        loss_1 = loss_1 * weights
    return loss_1



class SigmoidLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(SigmoidLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return sigmoid_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return sigmoid_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return sigmoid_with_weights(input, target, weights)
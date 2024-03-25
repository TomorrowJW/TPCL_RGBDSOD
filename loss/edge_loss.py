import torch
import torch.nn.functional as F

def cross_entropy2d_edge(input, target, reduction='mean'):
    assert (input.size()) == target.size()
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha *pos + beta*neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

def dice_loss(predict_, target):
    predict = torch.sigmoid(predict_)
    smooth = 1
    p =2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0],-1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2+smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num/den

    return loss.mean()
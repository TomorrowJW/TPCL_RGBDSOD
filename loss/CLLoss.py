import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange

'''
定义对比损失的算法
'''
class CLoss_1(nn.Module):
    def __init__(self, size, temperature):
        super().__init__()
        self.size = size
        self.temperature = temperature

    def forward(self, output, label):  # output: [b,c,h,w], label:[b,1,256,256]
        batch_size = label.size()[0]
        assert label.size() == torch.Size([batch_size,1,256,256])

        output = F.interpolate(output, size=(self.size, self.size), mode='bilinear', align_corners=True)
        label = F.interpolate(label, size=(self.size, self.size), mode='nearest')

        with torch.no_grad():
            label = label.squeeze(1)
            label = label.view(label.size(0), -1, 1) #[b, h*w, 1]
            mask = torch.eq(label, label.transpose(1, 2))
            # delete diag elem
            mask = mask.type(torch.FloatTensor).cuda()
            diag_mask = torch.scatter(torch.ones([mask.size(-1), mask.size(-2)]), 1,
                                       torch.arange(mask.size(-1)).view(-1, 1), 0).cuda()
            diag_mask = diag_mask.unsqueeze(0).repeat(batch_size, 1, 1)
            mask = mask * diag_mask
        # compute logits
        output = output.view(output.size(0), output.size(1), -1)
        output = torch.transpose(output,1,2) #[b,h*w,c]
        output = F.normalize(output,p=2,dim=2)
        dot = torch.matmul(output, output.transpose(-1, -2))
        logits = torch.div(dot, self.temperature)
        #delete diag elem
        logits = logits * diag_mask
        #for numerical stability
        logits_max, _ = torch.max(logits, dim=2, keepdim=True)
        logits = logits - logits_max.detach()
        #computer log prob
        exp_logits = torch.exp(logits)
        #mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=2, keepdim=True)+1e-12)
        #in case that mask.sum(2) is zero
        mask_sum = mask.sum(dim=2)
        mask_sum = torch.where(mask_sum==0,torch.ones_like(mask_sum),mask_sum)
        #compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=2) / mask_sum.detach()

        loss = torch.sum(pos_logits,1)/(self.size*self.size)
        loss = torch.mean(loss)
        loss = -loss

        return loss

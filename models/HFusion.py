import torch
import torch.nn as nn
from .Align import Align, BasicConv2d, TransBasicConv2d
from .Cross_ViT import Cross_ViT
from .CBAM import CBAM, SpatialAttention, ChannelAttention

class HF(nn.Module):
    def __init__(self, in_channel, dim, depth, heads, mlp_dim, num,
                             dim_head, dropout, emb_dropout, p1=4, p2=4, h=8, w=8):
        super(HF, self).__init__()
        self.rgb_b1 = BasicConv2d(in_channel, int(in_channel / 4), 3,padding=1, dilation=1)
        self.rgb_b2 = BasicConv2d(in_channel, int(in_channel / 4), 3, padding=3, dilation=3)
        self.rgb_b3 = BasicConv2d(in_channel, int(in_channel / 4), 3, padding=5, dilation=5)
        self.rgb_b4 = BasicConv2d(in_channel, int(in_channel / 4), 3, padding=7, dilation=7)

        self.d_b1 = BasicConv2d(in_channel, int(in_channel / 4), 3,padding=1, dilation=1)
        self.d_b2 = BasicConv2d(in_channel, int(in_channel / 4), 3, padding=3, dilation=3)
        self.d_b3 = BasicConv2d(in_channel, int(in_channel / 4), 3, padding=5, dilation=5)
        self.d_b4 = BasicConv2d(in_channel, int(in_channel / 4), 3, padding=7, dilation=7)

        self.cv1 = Cross_ViT(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, num=num,
                             dim_head = dim_head, dropout = dropout, emb_dropout = emb_dropout, p1=p1, p2=p2, h=h, w=w)
        self.cv2 = Cross_ViT(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, num=num,
                             dim_head = dim_head, dropout = dropout, emb_dropout = emb_dropout, p1=p1, p2=p2, h=h, w=w)
        self.cv3 = Cross_ViT(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, num=num,
                             dim_head = dim_head, dropout = dropout, emb_dropout = emb_dropout, p1=p1, p2=p2, h=h, w=w)
        self.cv4 = Cross_ViT(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, num=num,
                             dim_head = dim_head, dropout = dropout, emb_dropout = emb_dropout, p1=p1, p2=p2, h=h, w=w)

        self.ca1 = ChannelAttention(in_planes=int(in_channel/4), ratio=4)
        self.sa1 = SpatialAttention()

        self.ca2 = ChannelAttention(in_planes=int(in_channel/4), ratio=4)
        self.sa2 = SpatialAttention()

        self.ca3 = ChannelAttention(in_planes=int(in_channel/4), ratio=4)
        self.sa3 = SpatialAttention()

        self.ca4 = ChannelAttention(in_planes=int(in_channel/4), ratio=4)
        self.sa4 = SpatialAttention()

        self.conv = BasicConv2d(in_channel, in_channel*4,1,stride=1, padding=0, dilation=1)
    def forward(self,rgb,depth):

        f1_1, f2_1 = self.cv1(self.rgb_b1(rgb), self.d_b1(depth))
        f1 = f1_1 + f2_1
        f1= f1 + f1*self.sa1(f1*self.ca1(f1))

        f2_1, f2_2 = self.cv2(self.rgb_b2(rgb)+f1, self.d_b2(depth))
        f2 = f2_1 + f2_2
        f2= f2 + f2*self.sa2(f2*self.ca2(f2))

        f3_1, f3_2 = self.cv3(self.rgb_b3(rgb)+f2, self.d_b3(depth))
        f3 = f3_1 + f3_2
        f3= f3 + f3*self.sa3(f3*self.ca3(f3))

        f4_1, f4_2 = self.cv4(self.rgb_b4(rgb)+f3, self.d_b2(depth))
        f4 = f4_1 + f4_2
        f4= f4 + f4*self.sa4(f4*self.ca4(f4))

        f = torch.cat((f1,f2,f3,f4),1)

        f = self.conv(f)

        return f


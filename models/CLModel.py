import torch
import torch.nn as nn
from .Align import Align, BasicConv2d, TransBasicConv2d
from .Cross_ViT import Cross_ViT
from .Swin_V2 import SwinTransformerV2
from .depth_backbone import Depth_model
from .HFusion import HF
import math

class Edge_Aware(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.FFM_1 = nn.Sequential(BasicConv2d(256, 256, kernel_size=1, stride=1, padding=0),self.upsample)
        self.FFM_2 = nn.Sequential(BasicConv2d(256, 256, kernel_size=1, stride=1, padding=0),self.upsample,self.upsample)
        self.FFM1 = nn.Sequential(BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1), self.upsample)
        self.FFM2 = nn.Sequential(BasicConv2d(256, 1, kernel_size=1, stride=1, padding=0), self.upsample)

    def forward(self, F1, F2):
        F1_ = self.FFM_1(F1)
        F2_ = self.FFM_2(F2)
        #print(F1.size(),F2.size())
        F = torch.cat((F1_, F2_),1)
        F = self.FFM1(F)
        F = self.FFM2(F)

        return F

'''
定义decoder
'''
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4= nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.edge = Edge_Aware()
        self.FFM_1_0 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                 BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),self.upsample)
                                 
        self.FFM_1_1 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), self.upsample)

        self.FFM_2_0 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),self.upsample)
        self.FFM_2_1 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),self.upsample)

        self.FFM_S3 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),self.upsample)

        self.FFM_S2 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),self.upsample)

        self.S1 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),self.upsample)

        self.FFM_S1 = nn.Sequential(BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),self.upsample)

        self.c3 = nn.ModuleList([BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                 nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)])
        self.c2 = nn.ModuleList([BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                 nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)])
        self.c1 = nn.ModuleList([BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                 nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)])

        self.sigmoid = nn.Sigmoid()

    def forward(self, f1, f2, f3, f4):
        F_edge = self.edge(f1, f4)

        f4_3 = self.FFM_1_0(f4) + self.FFM_1_1(f3)
        F_S3 = self.upsample(f4_3)
        F_S3_CL = self.upsample(self.FFM_S3(F_S3))
        F_S3_CL = F_S3_CL + self.sigmoid(F_edge) * F_S3_CL
        F_S3_CL = self.c3[0](F_S3_CL)
        F_S3_CE = self.c3[1](F_S3_CL)

        f4_3_2 = self.FFM_2_0(f4_3) + self.FFM_2_1(f2)
        F_S2 = self.upsample(f4_3_2)
        F_S2_CL = self.FFM_S2(F_S2)
        F_S2_CL = F_S2_CL + self.sigmoid(F_edge) * F_S2_CL
        F_S2_CL = self.c2[0](F_S2_CL)
        F_S2_CE = self.c2[1](F_S2_CL)

        F_S1 = f4_3_2 + self.S1(f1)
        F_S1 = self.upsample(F_S1)
        F_S1_CL = self.FFM_S1(F_S1)
        F_S1_CL = F_S1_CL + self.sigmoid(F_edge) * F_S1_CL
        F_S1_CL = self.c1[0](F_S1_CL)
        F_S1_CE = self.c1[1](F_S1_CL)

        return F_S1_CE, F_S2_CE, F_S3_CE, F_edge, F_S1_CL, F_S2_CL, F_S3_CL

class CL_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swin_rgb = SwinTransformerV2()
        self.depth_model = Depth_model()

        self.Align = Align()

        self.HF1 = HF(64, config.dim_1, config.depth, config.heads, config.mlp_dim, config.num_1, \
                    dim_head=config.dim_head, dropout=config.dropout, emb_dropout=config.emb_dropout, p1=4, p2=4, h=8, w=8)
        self.HF2 = HF(64,config.dim_2, config.depth, config.heads, config.mlp_dim, config.num_2, \
                    dim_head=config.dim_head, dropout=config.dropout, emb_dropout=config.emb_dropout, p1=4, p2=4, h=4, w=4)

        self.decoder = Decoder()

        if self.training:
            self.initialize_weights()

    def forward(self, rgb, depth):  # label:[b,1,256,256]
        stage_rgb = self.swin_rgb(rgb)
        stage_depth = self.depth_model(depth)

        stage_rgb,stage_depth = self.Align(stage_rgb, stage_depth)

        rgb_1, d_1 = stage_rgb[0], stage_depth[0]
        rgb_2, d_2 = stage_rgb[1], stage_depth[1]
        rgb_3, d_3 = stage_rgb[2], stage_depth[2]
        rgb_4, d_4 = stage_rgb[3], stage_depth[3]

        f_1 = self.HF1(rgb_1, d_1)
        f_2 = self.HF1(rgb_2, d_2)
        f_3 = self.HF2(rgb_3, d_3)
        f_4 = self.HF2(rgb_4, d_4)

        F_S1_CE, F_S2_CE, F_S3_CE, F_edge, F_S1_CL, F_S2_CL, F_S3_CL = self.decoder(f_1,f_2,f_3,f_4)
        return F_S1_CE, F_S2_CE, F_S3_CE, F_edge, F_S1_CL, F_S2_CL, F_S3_CL

    def initialize_weights(self):  # 加载预训练模型权重，做初始化
        self.swin_rgb.load_state_dict(
             torch.load('/home/jasonwu/1-projects/2-RGB-D/pre_train/swinv2_base_patch4_window16_256.pth')['model'],
             strict=False)



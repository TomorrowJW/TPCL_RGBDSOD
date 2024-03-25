import torch
from torch import nn
from .Align import Align, BasicConv2d, TransBasicConv2d

class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv,self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=in_plane,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=in_plane)

        self.point_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=out_plane,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

        self.bn = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU(out_plane)

    def forward(self,x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Depth_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(BasicConv2d(1,128,1,stride=1, padding=0, dilation=1),
                                       BasicConv2d(128,128,3,stride=2, padding=1, dilation=1),
                                       BasicConv2d(128,128,3,stride=2, padding=1, dilation=1))

        self.stage_1 = DWConv(128, 256)
        self.stage_2 = DWConv(256, 512)
        self.stage_3 = DWConv(512, 1024)
        self.stage_4 = BasicConv2d(1024, 1024,1,stride=1, padding=0, dilation=1)


    def forward(self, depth): #[b,1,256,256]
        batch_size = depth[0]
        init = self.init_conv(depth) 
        stage_1 = self.stage_1(init) 
        stage_2 = self.stage_2(stage_1)
        stage_3 = self.stage_3(stage_2)
        stage_4 = self.stage_4(stage_3)

        return init,stage_1,stage_2,stage_3,stage_4
















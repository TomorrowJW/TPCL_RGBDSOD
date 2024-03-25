import torch.nn as nn

'''
定义基本卷积
'''


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


'''
定义转置卷积
'''


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1,
                 bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


'''
将各个阶段特征维度进行对齐，方便送入ViT模块
'''
class Align(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #Swinv2_Base_w16
        self.rgb_1 = BasicConv2d(256, 64, kernel_size=1, stride=1, padding=0) #[256,32,32] --> [64,32,32]
        self.rgb_2 = nn.Sequential(BasicConv2d(512, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[512,16,16] --> [64,32,32]
        self.rgb_3 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample) #[1024,8,8] --> [64,16,16]
        self.rgb_4 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[1024,8,8] --> [64,16,16]

        #Swinv2-Tiny
        #self.rgb_1 = BasicConv2d(192, 64, kernel_size=1, stride=1, padding=0) #[192,32,32] --> [64,32,32]
        #self.rgb_2 = nn.Sequential(BasicConv2d(384, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[384,16,16] --> [64,32,32]
        #self.rgb_3 = nn.Sequential(BasicConv2d(768, 64,kernel_size=1, stride=1, padding=0), self.upsample) #[768,8,8] --> [64,16,16]
        #self.rgb_4 = nn.Sequential(BasicConv2d(768, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[768,8,8] --> [64,16,16]

        #ResNet
        #self.rgb_1 = BasicConv2d(256, 64, kernel_size=3, stride=2, padding=1) #[256,64,64] --> [64,32,32]
        #self.rgb_2 = nn.Sequential(BasicConv2d(512, 64,kernel_size=1, stride=1, padding=0))#[512,32,32] --> [64,32,32]
        #self.rgb_3 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0)) #[1024,16,16] --> [64,16,16]
        #self.rgb_4 = nn.Sequential(BasicConv2d(2048, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[2048,8,8] --> [64,16,16]

        #PvT_v2
        #self.rgb_1 = BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1)  # [64,64,64] --> [64,32,32]
        #self.rgb_2 = nn.Sequential(BasicConv2d(128, 64, kernel_size=1, stride=1, padding=0))  # [128,32,32] --> [64,32,32]
        #self.rgb_3 = nn.Sequential(BasicConv2d(320, 64, kernel_size=1, stride=1, padding=0))  # [320,16,16] --> [64,16,16]
        #self.rgb_4 = nn.Sequential(BasicConv2d(512, 64, kernel_size=1, stride=1, padding=0),self.upsample)  # [512,8,8] --> [64,16,16]

        #P2T
        #self.rgb_1 = BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1)  # [64,64,64] --> [64,32,32]
        #self.rgb_2 = nn.Sequential(BasicConv2d(128, 64, kernel_size=1, stride=1, padding=0))  # [128,32,32] --> [64,32,32]
        #self.rgb_3 = nn.Sequential(BasicConv2d(320, 64, kernel_size=1, stride=1, padding=0))  # [320,16,16] --> [64,16,16]
        #self.rgb_4 = nn.Sequential(BasicConv2d(512, 64, kernel_size=1, stride=1, padding=0),self.upsample)  # [512,8,8] --> [64,16,16]

        #VGG16
        # self.rgb_1 = nn.Sequential(BasicConv2d(128, 64, kernel_size=3, stride=2, padding=1),
        #                            BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1))#[128,128,128] --> [64,32,32]
        # self.rgb_2 = nn.Sequential(BasicConv2d(256, 64,kernel_size=3, stride=2, padding=1))#[256,64,64] --> [64,32,32]
        # self.rgb_3 = nn.Sequential(BasicConv2d(512, 64,kernel_size=3, stride=2, padding=1)) #[512,32,32] --> [64,16,16]
        # self.rgb_4 = nn.Sequential(BasicConv2d(512, 64,kernel_size=1, stride=1, padding=0))#[512,16,16] --> [64,16,16]

        #T2T_vit_t_14
        #self.rgb_1 = BasicConv2d(64, 64, kernel_size=1, stride=1, padding=0) #[64,56,56] --> [64,56,56]
        #self.rgb_2 = nn.Sequential(BasicConv2d(64, 64, kernel_size=1, stride=1, padding=0),self.upsample) #[64,28,28] --> [64,56,56]
        #self.rgb_3 = nn.Sequential(BasicConv2d(384, 64,kernel_size=1, stride=1, padding=0)) #[384,14,14] --> [64,14,14]
        #self.rgb_4 = nn.Sequential(BasicConv2d(384, 64,kernel_size=1, stride=1, padding=0)) #[384,14,14] --> [64,14,14]

        #-----------------------------------------------------------------------------------------------------------------------

        #PVTv2

        #Depth_Model
        self.d_1 = BasicConv2d(256, 64, kernel_size=1, stride=1, padding=0) #[256,32,32] --> [64,32,32]
        self.d_2 = nn.Sequential(BasicConv2d(512, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[512,16,16] --> [64,32,32]
        self.d_3 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample) #[1024,8,8] --> [64,16,16]
        self.d_4 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[1024,8,8] --> [64,16,16]

        #ResNet
        #self.d_1 = BasicConv2d(256, 64, kernel_size=3, stride=2, padding=1) #[256,64,64] --> [64,32,32]
        #self.d_2 = nn.Sequential(BasicConv2d(512, 64,kernel_size=1, stride=1, padding=0))#[512,32,32] --> [64,32,32]
        #self.d_3 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0)) #[1024,16,16] --> [64,16,16]
        #self.d_4 = nn.Sequential(BasicConv2d(2048, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[2048,8,8] --> [64,16,16]

        #Swinv2
        #self.d_1 = BasicConv2d(256, 64, kernel_size=1, stride=1, padding=0) #[256,32,32] --> [64,32,32]
        #self.d_2 = nn.Sequential(BasicConv2d(512, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[512,16,16] --> [64,32,32]
        #self.d_3 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample) #[1024,8,8] --> [64,16,16]
        #self.d_4 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[1024,8,8] --> [64,16,16]

        #T2T对应的Depth_Model
        #self.d_1 = nn.Sequential(BasicConv2d(256, 64, kernel_size=1, stride=1, padding=0), self.upsample) #[256,32,32] --> [64,64,64]
        #self.d_2 = nn.Sequential(BasicConv2d(512, 64,kernel_size=1, stride=1, padding=0), self.upsample, self.upsample)#[512,16,16] --> [64,64,64]
        #self.d_3 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample) #[1024,8,8] --> [64,16,16]
        #self.d_4 = nn.Sequential(BasicConv2d(1024, 64,kernel_size=1, stride=1, padding=0), self.upsample)#[1024,8,8] --> [64,16,16]

    def forward(self,x,y):

        '''
        RGB转换
        '''
        rgb_1 = self.rgb_1(x[1])
        #print(rgb_1.size())
        #assert rgb_1.size(2) == 32
        rgb_2 = self.rgb_2(x[2])
        rgb_3 = self.rgb_3(x[3])
        rgb_4 = self.rgb_4(x[4])

        '''
        Depth转换
        '''
        d_1 = self.d_1(y[1])
        d_2 = self.d_2(y[2])
        #assert d_1.size(2) == 32
        d_3 = self.d_3(y[3])
        d_4 = self.d_4(y[4])

        
        return (rgb_1, rgb_2, rgb_3, rgb_4), (d_1, d_2, d_3, d_4)
        
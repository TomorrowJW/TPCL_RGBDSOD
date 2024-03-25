import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
#这里用了einops这个库
'''
Vision Transformer 的Encoder作为融合模块
'''

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
'''
0-定义标准化层，用的是Layer Norm (LN)
'''
class PreNorm(nn.Module):
    def __init__(self, dim, fn): #dim是归一化的维度也就是输入的维度，其实也就是第三维度，fn是函数
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn #具体的函数
    def forward(self, x, y): 
        return self.fn(self.norm(x), self.norm(y)) #先做Norm，然后做第二阶段

'''
1-定义Encoder第二阶段的前向传播层
'''
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.): #dim就是输入的维度,hidden_dim是隐藏层的维度,dropout是失活
        super().__init__()
        
        #就是一个两层的全连接层，不过激活函数采用的是GELU
        self.net_rgb = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.net_d = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, y): #x是rgb输入特征,y是d输入特征
        return x + self.net_rgb(x), y + self.net_d(y) #做全连接层且做残差
        
'''
2-定义Self-Attention层, 这里直接定义多头Self-Attention
'''
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.): #dim就是输入的维度，head是头的数目，dim_head是每个头的维度,其实也是Q、K、V的维度
        super().__init__()
        
        inner_dim = dim_head *  heads  # 头的维度乘以头的数目，先把总的变换维度得到，这里假如是64维度，64*8=512维
        project_out = not (heads == 1 and dim_head == dim) #如果头的数目不是1且头的维度不是等于输入维度, project_out=True

        self.heads = heads 
        self.scale = dim_head ** -0.5 #自注意力层的scale操作

        self.attend = nn.Softmax(dim = -1) #自注意力层的softmax归一化操作
        self.dropout = nn.Dropout(dropout) 

        self.to_qkv_rgb = nn.Linear(dim, inner_dim * 3, bias = False) #[RGB模态]乘以3是为了得到相应的Q,K,V; 也就是说得到了八个头的Q,K,V, 分为8组，输入是embedding的维度
        self.to_qkv_d = nn.Linear(dim, inner_dim * 3, bias = False) #[depth模态]
        
        self.to_out_rgb_d = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() #这一步直接是定义Encoder第一阶段在自注意力后加一个全连接层映射回原来的维度，方便做残差连接
        
        self.to_out_d_rgb = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x, y): #x是rgb输入特征，y是d输入特征
        qkv_rgb = self.to_qkv_rgb(x).chunk(3, dim = -1) #torch.chunk这个函数是分割tensor,这里沿着最后一个维度切出3个tensor出来.其实也就是切出q,k,v【多个头】，返回元组
        qkv_d = self.to_qkv_d(y).chunk(3, dim = -1)
        q_rgb, k_rgb, v_rgb = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_rgb) #这里就是对qkv的维度进行转换，转换成多个头的单个self_attention
                                                                                           #也就是说原来假如是[b=10, n=5, (h d) = 64*8],b表示样本数目，n表示每个样本长度
                                                                                           #(h d)表示每个样本的输入维度经过变换后多个头的Q、K、V的维度；现在维度为：
                                                                                           #[b,h,n,d]h表示头的数目，n表示样本长度，d表示映射后单个头的Q、K、V的维度
        q_d, k_d, v_d = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_d)
        
        '''
        做RGB对depth的Self_attention
        '''
        dots_rgb_d = torch.matmul(q_rgb, k_d.transpose(-1, -2)) * self.scale #做单个头的self-attention，并进行缩放操作

        attn_rgb_d = self.attend(dots_rgb_d) 
        attn_rgb_d = self.dropout(attn_rgb_d) #进一步得到softmax归一化后的权重

        out_rgb_d = torch.matmul(attn_rgb_d, v_d) #乘回v
        out_rgb_d = rearrange(out_rgb_d, 'b h n d -> b n (h d)') #将多个头拼接起来
        
        '''
        做depth对rgb的Self_attention
        '''
        dots_d_rgb = torch.matmul(q_d, k_rgb.transpose(-1, -2)) * self.scale #做单个头的self-attention，并进行缩放操作

        attn_d_rgb = self.attend(dots_d_rgb) 
        attn_d_rgb = self.dropout(attn_d_rgb) #进一步得到softmax归一化后的权重

        out_d_rgb = torch.matmul(attn_d_rgb, v_rgb) #乘回v
        out_d_rgb = rearrange(out_d_rgb, 'b h n d -> b n (h d)') #将多个头拼接起来
        
        
        
        return self.to_out_rgb_d(out_rgb_d) + x, self.to_out_d_rgb(out_d_rgb) + y  #最后映射回原来的维度

'''
3-定义Transformer-Encoder层
'''
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.): #dim就是输入的维度，depth表示encoder层的数目,mlp_dim=hidden_dim
                                                                            #表示第二阶段MLP层中间变换的维度
        super().__init__()
        
        self.layers = nn.ModuleList([]) #定义多层
        
        for _ in range(depth): #depth就是层数
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), #第一阶段
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)) #第二阶段
            ]))
            
    def forward(self, x, y):
        for attn, ff in self.layers: #多层前向传播
            x, y = attn(x, y)  #注意这里仍然是先加后做Norm，容易迷惑的地方在于PreNorm这个类，其实是一样的，先对输入做归一化才能输入到第二阶段
            x, y = ff(x, y) #第二阶段
        return x, y

'''
4-定义Cross_ViT的架构，注意这里没有Decoder架构
'''
class Cross_ViT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, num, dim_head = 64, dropout = 0., emb_dropout = 0., p1=4, p2=4, h=8, w=8):
        
        super().__init__()

        self.pos_embedding_rgb = nn.Parameter(torch.randn(1, num, dim)) #这里表示位置编码，这里用一个可学习的变量来替代之前的位置编码
        self.pos_embedding_d = nn.Parameter(torch.randn(1, num, dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.LN = nn.LayerNorm(dim)

        self.tran_rgb = Rearrange('b c (h p1) (w p2) -> b (h w ) (p1 p2 c)', p1=p1, p2=p2)
        self.tran_d = Rearrange('b c (h p1) (w p2) -> b (h w ) (p1 p2 c)', p1=p1, p2=p2)

        self.tran_1 = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c)->b c (h p1) (w p2)', h=h, w=w, p1=p1, p2=p2))

        self.tran_2 = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c)->b c (h p1) (w p2)', h=h, w=w, p1=p1, p2=p2))

    def forward(self, x0, y0): #传入x [b,num,dim]
        x = self.tran_rgb(x0)
        #print(x.size())
        y = self.tran_d(y0)

        x += self.pos_embedding_rgb #传入x与位置编码
        y += self.pos_embedding_d
        
        x = self.dropout(x) #embedding的dropout
        y = self.dropout(y)
        
        x, y = self.transformer(x,y)
        x = self.LN(x)
        y = self.LN(y)


        x = self.tran_1(x)
        y = self.tran_2(y)

        return x,y

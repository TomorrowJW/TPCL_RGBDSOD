import torch
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

#定义配置文件,写入一个配置类中
class Config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    Cross_ViT层的一些参数
    '''
    dim_1 = 256 #输入维度
    dim_2 = 256
    
    hidden_dim = 64 #前向层隐藏维度
    mlp_dim = 64 #Transformer前向层隐藏维度和hidden_dim一样大
    
    emb_dropout = 0.3 #嵌入层的dropout
    
    heads = 8 #attention的头的数目
    dim_head = 32 #attention的头的维度，也就是Q、K、V的维度
    dropout = 0.3 #attention层的dropout
    depth = 1 #Transformer的Encoder的层数 #T2t为4,其他为1
    
    num_1 = 64  #Cross_ViT输入样本的长度[底两层的长度]
    num_2 = 16  #Cross_ViT输入样本的长度[高两层的长度]

    temperature = 0.3 #对比学习温度值
    lr = 5e-5 #学习率
    weight_decay=1e-4
    num_epochs = 300
    a = 0.1 #这3个超参数用来调节损失比重
    b = 1
    c = 0.9
    clip = 0.5
    
    batch_size = 3
    train_size = 256
    test_size = 256
    num_workers = 18
    
    rgb_path = './Datasets/train_2985/train_images/' #训练rgb图片的路径
    d_path = './Datasets/train_2985/train_depth/' #训练depth图片的路径
    GT_path = './Datasets/train_2985/train_masks/' #训练标签的路径
    Edge_path = './Datasets/train_2985/train_edges/'#

    test_path = './Datasets//test_dataset/' #测试集的路径

    save_model_path = './Save_Models/'
    save_results_path = './Save_Results_2185/prediction/'
    save_edge_results_path = './Save_Results_2185/edges/'

    CL_size = 52



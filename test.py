import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import misc
from thop import profile
from thop import clever_format
import time
from models.CLModel import CL_Model
from Data_Loader import test_dataset
from config import Config
from tqdm import tqdm

cfg = Config()
dataset_path = cfg.test_path

model = CL_Model(cfg).to(cfg.device)
#Loads
model.load_state_dict(torch.load(cfg.save_model_path + '2985_L.pth'))
model.cuda()
model.eval()

# 计算参数量
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Parameters: {params:.3f} M")


test_datasets = ['STERE', 'NLPR', 'SIP', 'LFSD', 'DUT-RGBD', 'NJU2K','DES']

for dataset in test_datasets:
    save_path = cfg.save_results_path + dataset + '/'
    edge_save_path = cfg.save_edge_results_path + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs((edge_save_path))
    image_root = dataset_path + dataset + '/RGB/'
    depth_root = dataset_path + dataset + '/depth/'
    gt_root = dataset_path + dataset + '/GT/'

    test_loader = test_dataset(image_root, depth_root, gt_root, cfg.test_size)
    time_sum = 0
    for i in range(test_loader.size):
        image, depth, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(cfg.device)
        depth = depth.to(cfg.device)

        torch.cuda.synchronize()
        time_start = time.time()
        F_, F_S2_CE, F_S3_CE, F_edge, F_S1_CL, F_S2_CL, F_S3_CL = model(image, depth)
        torch.cuda.synchronize()
        time_end = time.time()
        time_sum = time_sum + (time_end - time_start)

        res = F.interpolate(F_, size=gt.shape, mode='bilinear', align_corners=False)
        edge = F.interpolate(F_edge, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
        cv2.imwrite(edge_save_path + name, edge * 255)
        if i == test_loader.size-1:
            print(dataset+' '+ 'is'+' '+'Running time {:.5f}'.format(time_sum/test_loader.size))
            print(dataset+' '+ 'is'+' '+'Average speed:{:.4f} fps'.format(test_loader.size/time_sum))




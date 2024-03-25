import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
import time
import torch.nn.functional as F
from datetime import datetime
import os
import torch.backends.cudnn as cudnn
from models.CLModel import CL_Model
from Data_Loader import get_loader
from utils import clip_gradient,warmup_poly,poly_lr
from config import Config
import matplotlib.pyplot as plt
from loss.CLLoss import CLoss_1
from loss.edge_loss import cross_entropy2d_edge,dice_loss
from loss.HEL_loss import HEL

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True

setup_seed(42)

cfg = Config()
model = CL_Model(cfg).to(cfg.device)
total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter : %.2fM'%(total/1e6))

train_dataloader = get_loader(cfg, batchsize=cfg.batch_size,\
                              trainsize=cfg.train_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
total_step = len(train_dataloader)

BCE_loss = nn.BCEWithLogitsLoss().cuda() 
IOU_loss = IOU(size_average=True).cuda()
HEL_loss = HEL().cuda()
CLoss1 = CLoss_1(cfg.CL_size, cfg.temperature).cuda()
CLoss2 = CLoss_1(cfg.CL_size, cfg.temperature).cuda()
CLoss3 = CLoss_1(cfg.CL_size, cfg.temperature).cuda()
params = model.parameters()
optimizer = torch.optim.AdamW(params, cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.weight_decay)

def train():
    print('Let us start to train the model:')
    sum_time = 0
    start_time = time.time()
    for epoch in range(cfg.num_epochs):
        poly_lr(optimizer,cfg.lr,epoch,cfg.num_epochs)
        model.train()
         
        for i,data in enumerate(train_dataloader, start=1): 
            optimizer.zero_grad() 
            
            images, depths, gts, edges = data
            images = images.to(cfg.device)
            depths = depths.to(cfg.device)
            gts = gts.to(cfg.device)
            edges = edges.to(cfg.device)

            F_S1_CE, F_S2_CE, F_S3_CE, F_edge, F_S1_CL, F_S2_CL, F_S3_CL = model(images, depths)

            loss_ce = 1 * (BCE_loss(F_S1_CE, gts) + HEL_loss(F_S1_CE,gts)) + \
                      0.5 * (BCE_loss(F_S2_CE,gts) + HEL_loss(F_S2_CE,gts)) \
                        + 0.5 * (BCE_loss(F_S3_CE,gts)  + HEL_loss(F_S3_CE,gts))

            loss_ed = 1 * (cross_entropy2d_edge(F_edge,edges) + dice_loss(F_edge,edges))

            loss_cl = 1 * CLoss1(F_S1_CL,gts) + 0.5 * CLoss1(F_S2_CL,gts) \
                        + 0.5 * CLoss1(F_S3_CL,gts)


            loss = cfg.c * loss_cl + cfg.b * loss_ed +cfg.a * loss_ce
            loss.backward()

            clip_gradient(optimizer, cfg.clip)
            optimizer.step()

            if i % 100 == 0 or i == total_step:
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Lossce: {:.8f}, Lossed: {:.8f}, Losscl: {:.12f}, Loss:{:.8f}'.
                  format(epoch, cfg.num_epochs, i, total_step, optimizer.param_groups[0]['lr'], loss_ce.item(), loss_ed.item(), loss_cl.item(),loss.item()))

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), cfg.save_model_path + '.%d' % (epoch) + 'CL.pth')
    end_time = time.time()
    time_sum = sum_time+ (end_time - start_time)
    print(time_sum)

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('forkserver', force=True)
    #torch.multiprocessing.set_start_method('spawn')
    train()

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch

#几个数据增强的策略

def cv_random_flip(img, label, depth, edge): #随机翻转
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, depth, edge
    
def randomCrop(image, label, depth, edge): #随机裁剪
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region),edge.crop(random_region)
    
def randomRotation(image, label, depth, edge): #随机旋转
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image,label,depth,edge
    
def colorEnhance(image): #随机颜色增强
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
    
def randomGaussian(image, mean=0.1, sigma=0.35): #随机高斯
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
    
def randomPeper(img):
    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randX=random.randint(0,img.shape[0]-1)  
        randY=random.randint(0,img.shape[1]-1)  
        if random.randint(0,1)==0:  
            img[randX,randY]=0  
        else:  
            img[randX,randY]=255 
    return Image.fromarray(img)  
    
#构造用于训练集的数据类
class Train_Dataset(data.Dataset):
    def __init__(self, config, trainsize): #trainsize是指缩放的大小
    
        self.trainsize = trainsize
        self.imgae_path = config.rgb_path
        self.depth_path = config.d_path
        self.gt_path = config.GT_path
        self.edge_path =config.Edge_path

        self.images = [self.imgae_path + f for f in os.listdir(self.imgae_path) if f.endswith('.jpg')]
        self.gts = [self.gt_path + f for f in os.listdir(self.gt_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [self.depth_path + f for f in os.listdir(self.depth_path) if f.endswith('.png') or f.endswith('jpg') or f.endswith('.bmp')]
        self.edges = [self.edge_path + f for f in os.listdir(self.gt_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)

        self.filter_files()
        self.size = len(self.images) #获取训练集数量
        
        #图像预处理
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #RGB需要归一化

        self.dep_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index): #获取每一条样本，训练集可以进行一些数据增强策略
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        edge = self.binary_loader(self.edges[index])
        #
        # image_ori = self.img_transform(image)
        # depth_ori = self.dep_transform(depth)
        # gt_ori = self.gt_transform(gt)
        # edge_ori = self.edge_transform(edge)


        image,gt,depth,edge =cv_random_flip(image,gt,depth,edge)
        image,gt,depth,edge=randomCrop(image, gt, depth,edge)
        image,gt,depth,edge=randomRotation(image, gt,depth,edge)
        image=colorEnhance(image)
        gt=randomPeper(gt)

        image = self.img_transform(image)
        depth = self.dep_transform(depth)
        gt = self.gt_transform(gt)
        edge = self.edge_transform(edge)
        #print(gt.size())
        #print(edge.size())
        #gt_cl = torch.where(gt >0, torch.ones_like(gt, dtype=gt.dtype), gt)
        return image, depth, gt, edge#, image_ori, depth_ori, gt_ori, edge_ori
        
    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.images) == len(self.depths)
        images = []
        depths = []
        gts = []
        edges = []
        for img_path, dep_path, gt_path, edge_path in zip(self.images, self.depths, self.gts, self.edges): #将对应元素打包成元组
            img = Image.open(img_path)
            dep = Image.open(dep_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and gt.size == dep.size and edge.size == img.size:
                images.append(img_path)
                depths.append(dep_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.depths = depths
        self.gts = gts
        self.edges = edges
        
    def rgb_loader(self, path): #打开RGB路径，转化为RGB
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path): #打开depth路径，转化为灰度图
        with open(path, 'rb') as f:
            img = Image.open(f)
            #return img.convert('1')
            return img.convert('L')
            #return img
        
    def resize(self, img, gt, depth): #缩放大小
        assert img.size == gt.size and gt.size==depth.size #判断图像大小与GT大小是否一致
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h), Image.NEAREST) #重新缩放大小
        else:
            return img, gt, depth 

    def __len__(self):
        return self.size
        
#测试集和加载器
class test_dataset:
    def __init__(self, image_root, depth_root, gt_root, testsize):
    
        self.testsize = testsize
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png') or f.endswith('.bmp')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                       or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png') or f.endswith('.bmp')]
                       
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #RGB需要归一化
            
        self.dep_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
            
        self.gt_transform = transforms.ToTensor()
        
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        depth = self.binary_loader(self.depths[self.index])
        depth = self.dep_transform(depth).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def __len__(self):
        return self.size
 
#训练集加载器函数 
def get_loader(config, batchsize, trainsize, shuffle=True, num_workers=2, pin_memory=True):
    dataset = Train_Dataset(config, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader
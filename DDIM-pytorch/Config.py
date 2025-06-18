import torch
from torch import optim
from unet import UNet
from diffusion import DenoiseDiffusion
from datasets import Datasets

class Config:
    def __init__(self):
        self.timesteps = 1000  # 扩散一共需要的时间步
        self.eta = 0.0  # 控制随机性的参数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 运算设备
        self.batch_size = 128  # 一批输入数据的大小
        self.epoch = 200  # 训练轮次
        self.img_channels = 3  # 输入图像通道数
        self.channels = 128  # 将图像映射到的特征子空间数
        self.img_size = 32  # 输入图像的大小
        self.model = UNet(self.img_channels,self.channels)  # UNet模型
        self.diffusion = DenoiseDiffusion(self.model,self.timesteps,self.eta,device=self.device)  # 扩散模型
        self.train_dataset = Datasets(train=True)  # 训练集
        self.test_dataset = Datasets(train=False)  # 测试集
        self.optimizer = optim.Adam(self.diffusion.model.parameters(),lr=5e-5)  # 优化器

# class Config:
#     def __init__(self):
#         self.timesteps = 1000  # 扩散时间步
#         self.eta = 0.0         # 噪声随机性参数
#         self.batch_size = 128  # 批量大小
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.epoch = 30       # 总训练轮次
#         self.img_channels = 3  # 图像通道数
#         self.channels = 128     # UNet特征维度
#         self.img_size = 32     # 图像尺寸
#         self.train_dataset = Datasets(train=True,batch_size=self.batch_size)  # 训练集
#         self.test_dataset = Datasets(train=False,batch_size=self.batch_size)  # 测试集
#         self.max_norm = 1.0
        
#         # ---------------------- 学习率预热参数 ----------------------
#         self.lr = 2e-4       # 预热后的最大学习率
        
#         # ---------------------- 模型与优化器 ----------------------
#         self.model = UNet(self.img_channels, self.channels)
#         self.diffusion = DenoiseDiffusion(self.model, self.timesteps, self.eta)
        
#         # 初始化优化器（初始学习率设为预热起始值）
#         self.optimizer = optim.Adam(
#             self.diffusion.model.parameters(), 
#             lr=self.lr
#         )
        
#         # 定义学习率调度器
#         self.scheduler = optim.lr_scheduler.LambdaLR(
#             self.optimizer,
#             lr_lambda=lambda step:min(1.0,step/5000)
#         )


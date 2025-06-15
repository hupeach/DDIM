import jittor as jt
from jittor import optim
from unet import UNet
from diffusion import DenoiseDiffusion 
from datasets import Datasets

class Config:
    def __init__(self):
        self.timesteps = 1000  # 扩散时间步
        self.eta = 0.0         # 噪声随机性参数
        self.batch_size = 128  # 批量大小
        self.epoch = 100       # 总训练轮次
        self.img_channels = 3  # 图像通道数
        self.channels = 64     # UNet特征维度
        self.img_size = 32     # 图像尺寸
        self.train_dataset = Datasets(train=True,batch_size=self.batch_size)  # 训练集
        self.test_dataset = Datasets(train=False,batch_size=self.batch_size)  # 测试集
        
        # ---------------------- 学习率预热参数 ----------------------
        self.warmup_start_lr = 1e-6  # 预热起始学习率
        self.max_lr = 2e-4       # 预热后的最大学习率
        
        # ---------------------- 模型与优化器 ----------------------
        self.model = UNet(self.img_channels, self.channels)
        self.diffusion = DenoiseDiffusion(self.model, self.timesteps, self.eta, device=self.device)
        
        # 初始化优化器（初始学习率设为预热起始值）
        self.optimizer = optim.Adam(
            self.diffusion.model.parameters(), 
            lr=self.warmup_start_lr  # 初始学习率为预热起点
        )
        
        # 定义学习率调度器
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step:min(self.max_lr,step/5000)
        )



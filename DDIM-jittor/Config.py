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
        self.epoch = 200       # 总训练轮次
        self.img_channels = 3  # 图像通道数
        self.channels = 128     # UNet特征维度
        self.img_size = 32     # 图像尺寸
        self.train_dataset = Datasets(train=True,batch_size=self.batch_size)  # 训练集
        self.test_dataset = Datasets(train=False,batch_size=self.batch_size)  # 测试集
        self.model = UNet(self.img_channels, self.channels)
        self.diffusion = DenoiseDiffusion(self.model, self.timesteps, self.eta)
        self.lr = 1e-4  # 学习率
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.diffusion.model.parameters(), 
            lr=self.lr
        )



import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_beta_schedule
from utils import l2_loss
from utils import gather
import matplotlib.pyplot as plt

def compute_loss(truth,pred,type):
    """
    计算损失
    :param truth: 实际值
    :param pred: 预测值
    :param type: 计算损失的方法，分为l2距离和mse损失
    :return: 损失值
    """
    if type == 'l2':
        return l2_loss(truth,pred)
    elif type == 'mse':
        return F.mse_loss(truth,pred)
    else:
        raise ValueError("we just support type l2 and mse")

class DenoiseDiffusion:
    """
    一个工具类，用于封装加噪、采样、训练中的损失计算
    """
    def __init__(self,model:nn.Module,timesteps:int,eta:float,device:str=None):
        """
        初始化
        :param model: 生成噪声的模型
        :param timesteps: 采样过程中的时间步
        :param eta: 控制随机性的参数
        :param device: 运算设备
        """
        if device is None:
            device = 'cuda'

        self.model = model.to(device)
        self.beta = torch.tensor(get_beta_schedule('linear', timesteps),device=device,dtype=torch.float32)
        self.alpha = 1.-self.beta
        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        self.timesteps = timesteps
        self.eta = eta

    def compute_sigma(self,t:torch.Tensor,t_prev:torch.Tensor):
        """
        计算DDIM的sigma
        :param t: 时间
        :param t_prev:上一步的时间
        :return: 在采样步骤中时间t的sigma
        """
        alpha_now = self.alpha.gather(-1,t)
        alpha_prev = self.alpha.gather(-1,t_prev)
        return (self.eta * ((1.-alpha_prev) / (1.-alpha_now+1e-5))*(1.-alpha_now/(alpha_prev+1e-5))).reshape(-1,1,1,1)

    def q_sample(self,x0:torch.Tensor,t:torch.Tensor,noise:torch.Tensor=None):
        """
        将传入的图片x0转化成t时刻加噪的图片
        :param x0: 传入的图片
        :param t: 时间t
        :param noise: 可以不传，噪声，一般随机生成
        :return: x0经过t加噪后的xt
        """
        if noise is None:
            noise =torch.randn_like(x0,device=x0.device)
        const = gather(self.alpha_bar,t)
        mean = const**0.5*x0
        var = 1.-const
        return mean+var**0.5*noise

    def p_sample(self,xt:torch.Tensor,t:torch.Tensor,t_prev:torch.Tensor):
        """
        采样公式
        :param xt:加噪后的xt
        :param t: 当前的时间步
        :param t_prev: 上一时刻时间步
        :return: x_(t-1)
        """
        noise_g = self.model(xt,t)
        alpha_t = gather(self.alpha_bar,t)
        alpha_prev = gather(self.alpha_bar,t_prev)
        sigma = self.compute_sigma(t,t_prev)
        eps = torch.randn(xt.shape, device=xt.device)
        x0 = (xt - torch.sqrt(1.-alpha_t)*noise_g) / torch.sqrt(alpha_t)
        return torch.sqrt(alpha_prev)*x0+torch.sqrt(1.-alpha_prev-sigma**2)*noise_g+sigma*eps


    def loss(self,x0:torch.Tensor,noise:torch.Tensor=None):
        """
        用于计算训练过程的损失
        :param x0: 输入图片
        :param noise: 可不传入，噪声，一般选择随机生成
        :return: 损失值
        """
        batch_size = x0.shape[0]
        # t = torch.randint(
        #             low=0, high=self.timesteps, size=(batch_size // 2 + 1,), device=x0.device
        #         )
        # t = torch.cat([t, self.timesteps - t - 1], dim=0)[:batch_size]
        t = torch.randint(0,self.timesteps,(batch_size,),dtype=torch.long,device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0,device=x0.device)
        xt = self.q_sample(x0,t,noise)
        self.model.to(x0.device)
        noise_g = self.model(xt,t)
        return compute_loss(noise,noise_g,type='mse')


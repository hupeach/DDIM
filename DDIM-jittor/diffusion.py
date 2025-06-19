import jittor as jt
from jittor import nn
from jittor import Module
from utils import get_beta_schedule
from utils import l2_loss
from utils import gather
from unet import UNet


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
        return nn.mse_loss(truth,pred)
    else:
        raise ValueError("we just support type l2 and mse")
    
class DenoiseDiffusion:
    def __init__(self,model:Module,timesteps:int,eta:float):
        """
        初始化
        :param model: 生成噪声的模型
        :param timesteps: 采样过程中的时间步
        :param eta: 控制随机性的参数
        """
        self.model = model
        self.beta = jt.array(get_beta_schedule('linear', timesteps),dtype=jt.float32)
        self.alpha = 1.-self.beta
        self.alpha_bar = jt.cumprod(self.alpha,dim=0)
        self.timesteps = timesteps
        self.eta = eta

    def compute_sigma(self,t:jt.Var,t_prev:jt.Var):
        """
        计算DDIM的sigma
        :param t: 时间
        :param t_prev:上一步的时间
        :return: 在采样步骤中时间t的sigma
        """
        alpha_now = self.alpha.gather(-1,t)
        alpha_prev = self.alpha.gather(-1,t_prev)
        return (self.eta * ((1.-alpha_prev) / (1.-alpha_now+1e-5))*(1.-alpha_now/(alpha_prev+1e-5))).reshape(-1,1,1,1)

    def q_sample(self,x0:jt.Var,t:jt.Var,noise:jt.Var=None):
        """
        将传入的图片x0转化成t时刻加噪的图片
        :param x0: 传入的图片
        :param t: 时间t
        :param noise: 可以不传，噪声，一般随机生成
        :return: x0经过t加噪后的xt
        """
        if noise is None:
            noise =jt.randn_like(x0)
        const = gather(self.alpha_bar,t)
        mean = jt.sqrt(const)*x0
        var = 1.-const
        return mean+jt.sqrt(var)*noise

    def p_sample(self,xt:jt.Var,t:jt.Var,t_prev:jt.Var):
        """
        采样公式
        :param xt:加噪后的xt
        :param t: t
        :param t_prev: t-1
        :return: x_(t-1)
        """
        noise_g = self.model(xt,t)
        alpha_t = gather(self.alpha_bar,t)
        alpha_prev = gather(self.alpha_bar,t_prev)
        sigma = self.compute_sigma(t,t_prev)
        eps = jt.randn(xt.shape)
        x0 = (xt - jt.sqrt(1.-alpha_t)*noise_g) / jt.sqrt(alpha_t)
        return jt.sqrt(alpha_prev)*x0+jt.sqrt(1.-alpha_prev-sigma**2)*noise_g+sigma*eps


    def loss(self,x0:jt.Var,noise:jt.Var=None):
        """
        用于计算训练过程的损失
        :param x0: 输入图片
        :param noise: 可不传入，噪声，一般选择随机生成
        :return: 损失值
        """
        batch_size = x0.shape[0]
        # t = jt.randint(low=0, high=self.timesteps, shape=(batch_size // 2 + 1,))
        # t = jt.concat([t, self.timesteps - t - 1], dim=0)[:batch_size]
        t = jt.randint(0,self.timesteps,(batch_size,),dtype=jt.int32)
        if noise is None:
            noise = jt.randn_like(x0)
        xt = self.q_sample(x0,t,noise)
        noise_g = self.model(xt,t)
        loss = compute_loss(noise,noise_g,type='mse')
        return loss.float32()




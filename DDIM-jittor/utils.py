import math
import numpy as np
import jittor as jt
import matplotlib.pyplot as plt


def betas_for_alpha_bar(t:int, alpha_bar, max_beta:float=0.999):
    """
    余弦调度生成函数，数值更加稳定平滑，使得噪声前期增长慢，后期增长快，同时防止后期过大
    :param t: 训练过程需要的时间步
    :param alpha_bar: 一个匿名函数，用于计算time步后扩散后保留原始数据的概率
    :param max_beta: 最大的beta值，防止超过题目给出的限制
    """
    betas = []
    for i in range(t):
        t1 = i / t
        t2 = (i + 1) / t
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_beta_schedule(name:str, t:int):
    """
    获得beta时间表，满足0<beta<1
    :param name: 包括linear和cosine
    :param t: 训练过程需要的时间步
    :return:beta参数表，一维np形式，包含t个元素
    """
    if name == "linear":
        # 线性获取
        scale = 1000 / t
        beta_start = scale * 0.0001
        beta_end = min(scale * 0.02,0.999)
        return np.linspace(
            beta_start, beta_end, t, dtype=np.float64
        )
    elif name == "cosine":
        # 余弦调度获取
        return betas_for_alpha_bar(
            t,
            lambda t1: math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")

def l2_loss(truth:jt.array, pred:jt.array):
    """
    计算l2损失
    :param truth:实际值
    :param pred: 预测值
    :return: l2损失
    """
    return jt.sum((pred-truth)**2,dims=(1,2,3)).mean()

def gather(consts: jt.array, t: jt.array):
    """
    从consts取出第t个元素并重塑形状
    :param consts: 参数表
    :param t: 位置
    :return: 重塑后的可运算的向量
    """
    c = consts.gather(-1, t)  # 从最后一个维度取出第t个元素
    return c.reshape(-1, 1, 1, 1)  # 重塑特征图形状，便于广播运算

def get_time_schedule(name:str,timesteps:int,n_steps:int):
    """
    获取DDIM采样过程需要使用的时间步
    :param name: 类型，包括linear、quad、cosine
    :param timesteps: 扩散过程中一共使用的时间步
    :param n_steps: 需要几步才能采样成功
    :return: 时间张量
    """
    if name == 'linear':
        # 线性采集，逆序输出
        t = jt.linspace(0,timesteps-1,n_steps)
        return t.flip(0)
    elif name == 'quad':
        i = jt.arange(n_steps, dtype=jt.float32)
        # 计算平方根反比分布,平方采集
        ratio = 1 - jt.sqrt(i / (n_steps - 1))
        t = timesteps * ratio
        # 截断到[0, T]并取整
        t = jt.clamp(t,0,timesteps-1).round().int32()
        return t
    elif name == 'cosine':
        i = jt.arange(n_steps, dtype=jt.float32)
        # 余弦分布采集
        ratio = (1 + jt.cos(jt.array(math.pi) * i / (n_steps - 1))) / 2
        t = timesteps * ratio
        t = jt.clamp(t, 0, timesteps-1).long()
        return t
    else:
        raise ValueError("we just support linear, quad and cosine")


def visualize(images,batch_size:int=16,save_path:str=None,dpi:int=300):
    """
    将批次图像拼接为 b*b 网格图
    :param images: 输入张量 [B, C, H, W]，值域为[-1, 1]
    :param batch_size: b = batch_size**0.5
    :param save_path: 保存路径（None 则显示图像）
    :param dpi: 图像分辨率
    """
    # 张量转 numpy
    if isinstance(images, jt.Var):
        images = images.detach().cpu().numpy()  # 转到 CPU 并转为 numpy
        images = np.transpose(images, (0, 2, 3, 1))  # [B, H, W, C]
    # 归一化还原
    images = (images+1)/2*255  # [-1, 1] -> [0, 255]
    images = images.astype(np.uint8)  # 转为整数格式
    # 创建 4×4 子图网格
    plt.figure(figsize=(4, 4), dpi=dpi)
    b = int(batch_size**0.5)
    for i in range(batch_size):
        plt.subplot(b,b,i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    # 优化布局并保存/显示
    plt.tight_layout(pad=0.1)  # 子图间距
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    else:
        plt.show()
    plt.close()
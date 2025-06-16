import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import tqdm
import math

class TimeEmbedding(nn.Module):
    """
    时间嵌入类，将输入的时间转化成向量
    相比于DDIM源码的直接使用公式计算，DDPM经过计算后再使用MLP能够更好捕获非线性时间的关系
    这里是一个计算量和性能的trade off
    """
    def __init__(self, n_channels: int):
        """
        初始化
        :param n_channels:生成时间向量的维度
        """
        super().__init__()
        self.n_channels = n_channels
        self.layer1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.swish = nn.SiLU()
        self.layer2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        """
        传入时间，输出嵌入向量
        :param t: 传入的时间
        :return: 嵌入的时间向量
        """
        # 使用公式计算时间向量
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # 使用MLP,存在可学习的参数，增强模型表达能力
        emb = self.layer1(emb)
        emb = self.swish(emb)
        emb = self.layer2(emb)
        return emb



class ResidualBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channels:int,
                 out_channels:int,
                 emb_channels:int,
                 n_groups:int=32,
                 conv_shortcut:bool=True,
                 dropout:float=0.1):
        """
        初始化
        :param in_channels: 输入的图片通道
        :param out_channels: 输出的图片通道
        :param emb_channels: 嵌入时间的维度
        :param n_groups: 组归一化的组数
        :param conv_shortcut:对于shortcut是否采用特征提取,默认True
        :param dropout:dropout采用的参数,默认0.1
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.GroupNorm(num_groups=n_groups,num_channels=in_channels)
        self.swish1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

        self.act_t = nn.SiLU()
        self.switch_dim = nn.Linear(emb_channels, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=n_groups,num_channels=out_channels)
        self.swish2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)

        if in_channels != out_channels:
            if conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x:torch.Tensor, t:torch.Tensor):
        """
        前向传播
        :param x: 网络的输入向量
        :param t: 时间信息
        :return: 提取出来的特征
        """
        h = x
        # 第一个卷积层
        h = self.norm1(h)
        h = self.swish1(h)
        h = self.conv1(h)
        # 将时间向量嵌入
        t = self.act_t(t)
        t = self.switch_dim(t)
        h += t[:,:,None,None]
        # 第二个卷积层，同时包括dropout
        h = self.norm2(h)
        h = self.swish2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        # 残差连接
        x = self.shortcut(x)
        return x+h


class AttentionBlock(nn.Module):
    """
    transformer的自注意力机制
    """
    def __init__(self, channels:int, n_groups:int=32):
        """
        初始化
        :param channels:输入通道数
        :param n_groups: 组归一化的组数
        """
        super().__init__()
        self.norm = nn.GroupNorm(n_groups,channels)
        self.q = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        self.k = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        self.v = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        self.out = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)


    def forward(self,x:torch.Tensor):
        """
        前向传播
        :param x: 需要关注的向量
        :return:经过关注的向量
        """
        h_ = x
        # 将x归一化映射到qkv特征子空间
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        # 计算自注意力分数
        b,c,w,h = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k)
        w_ = w_*(int(c)*-0.5)
        w_ = nn.functional.softmax(w_, dim=2)
        # 得到价值
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w)

        # 残差连接
        x = self.out(x)
        return h_+x

class DownBlock(nn.Module):
    """
    下采样过程中的层内逻辑
    由一个残差块和一个注意力组成，用于特征提取
    把in_channels变成out_channels
    """
    def __init__(self,in_channels:int,out_channels:int,emb_channels:int,has_attention:bool,n_groups:int=32):
        """
        每一层中的一个Block初始化
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param emb_channels: 时间嵌入维度
        :param has_attention: 是否存在注意力
        :param n_groups: 组归一化的组数
        """
        super().__init__()
        self.res = ResidualBlock(in_channels,out_channels,emb_channels,n_groups=n_groups)
        if has_attention:
            self.attention = AttentionBlock(out_channels,n_groups=n_groups)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x:torch.Tensor, t:torch.Tensor):
        """
        前向传播
        :param x: 输入向量
        :param t: 时间向量
        :return:unet的中间特征图 
        """
        x = self.res(x,t)
        x = self.attention(x)
        return x
    

class UpBlock(nn.Module):
    """
    上采样过程中的层内逻辑
    残差块加注意力
    由于跳跃连接存在，减少图片纹理丢失，与DownBlock不同之处就是与下采样过程同层的输出进行连接
    """
    def __init__(self,in_channels:int,out_channels:int,emb_channels:int,has_attention:bool,n_groups:int=32):
        """
        初始化
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param emb_channels: 时间维度
        :param has_attention: 是否存在注意力
        :param n_groups: 组归一化的组数
        """
        super().__init__()
        self.res = ResidualBlock(in_channels,out_channels,emb_channels,n_groups=n_groups)
        if has_attention:
            self.attention = AttentionBlock(out_channels,n_groups=n_groups)
        else:
            self.attention = nn.Identity()

    def forward(self,x:torch.Tensor,t:torch.Tensor):
        """
        前向传播
        :param x: 输入特征图
        :param t: 时间向量
        :return: 中间特征图
        """
        x = self.res(x,t)
        x = self.attention(x)
        return x


class MiddleBlock(nn.Module):
    """
    unet的中间部分逻辑
    两个残差加上注意力组成
    """
    def __init__(self,channels:int,emb_channels:int,n_groups:int=32):
        """
        初始化
        :param channels:输入通道
        :param emb_channels: 时间维度
        """
        super().__init__()
        self.res1 = ResidualBlock(channels,channels,emb_channels,n_groups=n_groups)
        self.attention = AttentionBlock(channels)
        self.res2 = ResidualBlock(channels,channels,emb_channels,n_groups=n_groups)

    def forward(self,x:torch.Tensor,t:torch.Tensor):
        """
        前向传播
        :param x:输入特征图
        :param t: 时间向量
        :return: 中间特征图
        """
        x = self.res1(x,t)
        x = self.attention(x)
        x = self.res2(x,t)
        return x


class DownSample(nn.Module):
    """
    下采样过程中的层间逻辑
    设置多种下采样方式：
    avg_pool:平均池化
    conv:卷积
    """
    def __init__(self,channels:int,mode:str='conv'):
        """
        初始化
        :param channels:输入通道
        :param mode: 采用下采样模式
        """
        super().__init__()
        self.mode = mode
        if mode == 'conv':
            self.conv = nn.Conv2d(channels,channels,kernel_size=3,stride=2,padding=0)

    def forward(self,x:torch.Tensor,t:torch.Tensor):
        """
        前向传播
        :param x: 输入特征图
        :param t: 无实际意义，与DownBlock保持一致
        :return: 输出特征图
        """
        if self.mode=='conv':
            pad = (0,1,0,1)
            x = nn.functional.pad(x,pad,'constant',0)
            x = self.conv(x)
        elif self.mode=='avg_pool':
            x = nn.functional.avg_pool2d(x,kernel_size=2,stride=2)
        else:
            raise ValueError("we just support conv and avg_pool")
        return x

class UpSample(nn.Module):
    """
    上采样过程的层间逻辑
    包括两种模式
    with_conv:插值+卷积
    no_conv:卷积
    """
    def __init__(self,channels,mode='with_conv'):
        super().__init__()
        self.mode = mode
        if 'conv' in mode:
            self.conv = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1)

    def forward(self,x:torch.Tensor,t:torch.Tensor):
        if 'conv' in self.mode:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            if self.mode == 'with_conv':
                x = self.conv(x)
        else:
            raise ValueError("we just support with_conv and no_conv")
        return x

class UNet(nn.Module):
    """
    U-Net代码,用于生成噪声
    """
    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 128,
        resolution: list = (1, 2, 2, 2),
        is_attn: list = (False, True, True, False),
        n_blocks: int = 2,
        n_groups: int = 32
    ):
        super().__init__()
        n_resolutions = len(resolution)
        emb_channels = n_channels * 4
        # 时间编码
        self.time_emb = TimeEmbedding(emb_channels)
        self.img_proj = nn.Conv2d(image_channels,n_channels,kernel_size=3,stride=1,padding=1)
        down = []
        in_channels = n_channels
        # 下采样逻辑
        for i in range(n_resolutions):
            # 层内逻辑
            out_channels = in_channels * resolution[i]
            for j in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, emb_channels, is_attn[i], n_groups=n_groups))
                in_channels = out_channels
            # 层间逻辑
            if i < n_resolutions - 1:
                down.append(DownSample(in_channels))
        self.down = nn.ModuleList(down)
        # 中间层
        self.middle = MiddleBlock(in_channels, emb_channels, n_groups=n_groups)
        up = []
        # 上采样逻辑
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            # 层内逻辑
            for j in range(n_blocks):
                up.append(UpBlock(in_channels + out_channels, out_channels, emb_channels, is_attn[i], n_groups=n_groups))
            out_channels = in_channels // resolution[i]
            # 跳跃连接
            up.append(UpBlock(in_channels + out_channels, out_channels, emb_channels, is_attn[i], n_groups=n_groups))
            in_channels = out_channels
            # 层间逻辑
            if i > 0:
                up.append(UpSample(in_channels))
        self.up = nn.ModuleList(up)
        # 对结果处理，恢复形状
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(n_channels, image_channels, 3, padding=1)
        
    
    def forward(self,x: torch.Tensor,t: torch.Tensor):
        t = self.time_emb(t)
        x = self.img_proj(x)
        value = [x]
        # 下采样，特征提取，并且记录每一个块的输出，用来跳跃连接
        for m in self.down:
            x = m(x, t)
            value.append(x)
        # 中间层，对更低维度的特征进行处理
        x = self.middle(x, t)
        # 上采样，跳跃连接，复原特征
        for m in self.up:
            if isinstance(m, UpSample) == False:
                s = value.pop()
                x = torch.cat([x,s], dim=1)
            x = m(x, t)
        # 恢复形状
        return self.final(self.act(self.norm(x)))


def test_unet_on_gpu():
    # -------------------- 环境准备 --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")
    assert device.type == "cuda", "请确保 GPU 可用！"

    # -------------------- 模型初始化 --------------------
    img_channels = 3
    channels = 32  # 与 TimeEmbedding 的 n_channels 一致
    model = UNet(
        img_channels=img_channels,
        channels=channels,
        resolution=[1, 2, 4],  # 调整为适合 16x16 输入的分辨率（示例用）
        is_attn=[False, True, False],  # 示例注意力配置
        n_blocks=1
    ).to(device)  # 模型移动到 GPU
    print("模型初始化完成，参数已移动到 GPU")

    # -------------------- 随机数据生成 --------------------
    batch_size = 16
    img_size = 16  # 输入图像尺寸（16x16）
    # 随机图像（形状：[batch, channels, height, width]）
    images = torch.randn(batch_size, img_channels, img_size, img_size).to(device)
    # 随机时间步（形状：[batch]，范围 0~100）
    t = torch.randint(0, 100, (batch_size,), dtype=torch.float32).to(device)

    # -------------------- 前向推理验证 --------------------
    with torch.no_grad():  # 关闭梯度（推理模式）
        outputs = model(images, t)

    # 验证输出形状是否正确（应与输入同尺寸）
    assert outputs.shape == images.shape, \
        f"输出形状错误！期望 {[batch_size, img_channels, img_size, img_size]}，实际 {outputs.shape}"
    print(f"前向推理成功！输出形状: {outputs.shape}（设备: {outputs.device}）")

    # -------------------- 训练循环模拟 --------------------
    print("\n开始模拟训练循环...")
    criterion = nn.MSELoss()  # 均方误差损失（示例用）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam 优化器

    # 单轮训练模拟
    for epoch in tqdm.tqdm(range(1), desc="训练轮次"):
        optimizer.zero_grad()  # 梯度清零

        # 前向传播
        outputs = model(images, t)

        # 计算损失
        fake_noise = outputs  # 模型输出的噪声
        real_noise = torch.randn_like(fake_noise)  # 模拟真实噪声
        loss = criterion(fake_noise, real_noise)

        # 反向传播与优化
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 打印训练信息
        print(f"训练损失: {loss.item():.6f}")
        print(f"模型参数已更新（GPU 上）")

    print("\n测试完成！模型在 GPU 上可正常推理和训练。")


# --------------------------
# 运行测试
# --------------------------
if __name__ == "__main__":
    test_unet_on_gpu()



import os
import glob
import numpy as np
from scipy.linalg import sqrtm
import shutil
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import inception_v3
from Config import Config
from DDIM import DDIM
from PIL import Image
from tqdm import tqdm
import argparse



def load_inception_model(device='cuda'):
    """
    加载预训练的 Inception v3 模型（仅保留特征提取层）
    :param device: 计算设备
    :return: Inception v3 模型
    """
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    # 移除最后全连接层（输出 1000 类），保留到 avgpool 层
    model.fc = torch.nn.Identity() 
    return model.to(device)

def extract_features(model, data_loader, device):
    """
    从数据加载器中提取 Inception v3 特征
    :param model: Inception v3 模型
    :param data_loader: 数据加载器
    :param device: 计算设备
    :return: 特征数组
    """
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取特征"):
            # 确保批次是 Tensor
            if isinstance(batch, list):  # 真实数据：(图像张量, 标签张量)
                images, _ = batch  # 提取图像部分
                batch = images
            if not isinstance(batch, torch.Tensor):
                raise TypeError(f"批次数据应为 Tensor，实际类型: {type(batch)}")
            
            # 检查张量维度
            if batch.dim() != 4:
                raise ValueError(f"输入张量维度错误：期望 4D，实际 {batch.dim()}D")
            
            batch = batch.to(device)  # 数据已为 [-1,1] 的 Tensor，形状 [B, C, H, W]
            
            # -------------------- 预处理：调整为 Inception 输入要求 --------------------
            # 步骤1：[-1,1] → [0,1]
            batch_0_1 = (batch + 1) / 2.0
            
            # 步骤2：调整大小到 299x299
            batch_resized = torch.nn.functional.interpolate(
                batch_0_1,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )  # 形状 [B, C, 299, 299]
            
            # 步骤3：中心裁剪到 299x299
            # 注意：interpolate 后尺寸可能略大于 299，需先裁剪再缩放
            batch_cropped = transforms.CenterCrop((299, 299))(batch_resized)  # 形状 [B, C, 299, 299]
            
            # 步骤4：归一化
            mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            batch_normalized = (batch_cropped - mean) / std  # 形状 [B, C, 299, 299]
            
            # -------------------- 提取 Inception 特征 --------------------
            # 输入 Tensor 范围 [0,1]，尺寸 299x299，符合 Inception 要求
            batch_features = model(batch_normalized)
            features.append(batch_features.cpu())
    
    # 合并所有特征为一个数组（形状 [N, 2048]）
    return torch.cat(features, dim=0).numpy()

def load_generated_samples(gen_dir, num_samples, img_size, device):
    """
    从 PNG 目录加载生成样本并预处理为模型输入格式
    :param gen_dir: 生成样本目录
    :param num_samples: 需要加载的样本数
    :param img_size: 图片尺寸
    :param device: 计算设备
    :return: 数据加载器
    """
    # 筛选并排序 PNG 文件
    gen_files = sorted(glob.glob(f"{gen_dir}/*.png"))
    if len(gen_files) < num_samples:
        raise ValueError(f"生成样本不足！需要 {num_samples} 张，实际找到 {len(gen_files)} 张")
    gen_files = gen_files[:num_samples]  # 截断到目标数量

    # 定义生成数据的预处理
    def gen_transform(img_pil):
        # 1. 转换为 [0,255] 的 NumPy 数组
        img_np = np.array(img_pil).astype(np.float32)
        # 2. 转换为 [-1,1] 范围
        img_np = (img_np / 255.0) * 2.0 - 1.0
        # 3. 转换为 [C, H, W] 张量
        return torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32)

    # 创建自定义数据集
    class GeneratedDataset(torch.utils.data.Dataset):
        def __init__(self, file_list, transform):
            self.file_list = file_list
            self.transform = transform

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            img_pil = Image.open(self.file_list[idx]).convert('RGB') 
            img_tensor = self.transform(img_pil)
            return img_tensor

    # 创建数据加载器
    gen_dataset = GeneratedDataset(gen_files, gen_transform)
    gen_loader = torch.utils.data.DataLoader(
        gen_dataset,
        batch_size=config.batch_size,  # 与训练时的 batch_size 一致
        shuffle=False,                 # 特征提取无需打乱顺序
        num_workers=4,                 # 数据加载线程数
        pin_memory=True                # 加速 GPU 内存拷贝
    )
    return gen_loader

def calculate_fid(
    gen_dir, 
    real_data_dir, 
    num_samples=50000, 
    img_size=32, 
    batch_size=256, 
    device='cuda'
):
    """
    计算生成样本与真实数据的 FID 分数
    :param gen_dir: 生成样本目录
    :param real_data_dir: 真实数据集目录
    :param num_samples: 计算使用的样本数
    :param img_size: 图片尺寸
    :param batch_size: 批次大小
    :param device: 计算设备
    :return: FID 分数
    """
    # -------------------- 初始化 Inception 模型 --------------------
    model = load_inception_model(device=device)

    # -------------------- 加载真实数据 --------------------
    print("加载真实数据集...")
    real_transform = transforms.Compose([
        transforms.Resize(img_size),          
        transforms.ToTensor(),               
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],             
            std=[0.5, 0.5, 0.5]
        )
    ])
    real_dataset = CIFAR10(
        root=real_data_dir,
        train=False,
        download=True,
        transform=real_transform
    )
    # 截断到目标样本数
    real_dataset = torch.utils.data.Subset(real_dataset, range(min(num_samples, len(real_dataset))))
    real_loader = torch.utils.data.DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True  # 加速 GPU 内存拷贝
    )
    print(f"真实数据加载完成，样本数: {len(real_dataset)}")

    # -------------------- 加载生成数据 --------------------
    print("加载生成数据...")
    try:
        gen_loader = load_generated_samples(
            gen_dir=gen_dir,
            num_samples=num_samples,
            img_size=img_size,
            device=device
        )
    except Exception as e:
        raise RuntimeError(f"生成数据加载失败: {str(e)}")
    print(f"生成数据加载完成，样本数: {num_samples}")

    # -------------------- 提取真实数据特征 --------------------
    print("提取真实数据特征...")
    try:
        real_features = extract_features(model, real_loader, device)
        print(f"真实数据特征形状: {real_features.shape}（期望 [{num_samples}, 2048]）")
    except Exception as e:
        raise RuntimeError(f"真实数据特征提取失败: {str(e)}")

    # -------------------- 提取生成数据特征 --------------------
    print("提取生成数据特征...")
    try:
        gen_features = extract_features(model, gen_loader, device)
        print(f"生成数据特征形状: {gen_features.shape}（期望 [{num_samples}, 2048]）")
    except Exception as e:
        raise RuntimeError(f"生成数据特征提取失败: {str(e)}")

    # -------------------- 计算 FID --------------------

    # 计算均值和协方差矩阵
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    # 计算协方差矩阵平方根
    cov_product = sigma_gen @ sigma_real
    try:
        cov_sqrt = sqrtm(cov_product)
    except np.linalg.LinAlgError:
        # 数值不稳定时添加小量
        cov_sqrt = sqrtm(cov_product + 1e-6 * np.eye(sigma_gen.shape[0]))

    # 处理复数情况
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # 计算 FID 分数
    diff = mu_gen - mu_real
    fid_score = np.dot(diff, diff) + np.trace(sigma_gen + sigma_real - 2 * cov_sqrt)
    return float(fid_score)


def safe_rmtree(directory, verbose=True):
    """
    安全删除目录及其所有内容
    :param directory: 要删除的目录路径
    :param verbose: 是否打印删除信息
    :return: 成功返回 True，失败返回 False
    """
    if not os.path.exists(directory):
        if verbose:
            print(f"目录不存在: {directory}（无需删除）")
        return True
    
    try:
        if verbose:
            print(f"正在删除目录: {directory}...")
        shutil.rmtree(directory)
        if verbose:
            print(f"目录删除成功: {directory}")
        return True
    except Exception as e:
        if verbose:
            print(f"删除目录失败: {directory}，错误原因: {str(e)}")
        return False

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='训练过程')
    parser.add_argument('-samples', type=int, default=10000, 
                        help='采样的样本数（默认：10000）')
    parser.add_argument('-steps', type=int, default=20, 
                        help='采样步数（默认：20）')
    parser.add_argument('-eta', type=float, default=0.0, 
                        help='控制随机性的参数（默认：0.0）')
    parser.add_argument('-BS', type=int, default=256, 
                        help='采样批次大小（默认：256）')
    args = parser.parse_args()
    # 初始化配置
    config = Config()
    config.batch_size = args.BS
    config.img_size = 32    
    config.eta = args.eta
    # 创建生成样本目录
    os.makedirs('generated_samples/fid_samples', exist_ok=True)
    
    # 初始化 DDIM 模型
    ddim = DDIM(config)
    
    # 生成样本
    ddim.sample_fid(total_n_samples=args.samples, batch_size=config.batch_size,n_steps=args.steps)
    
    # 计算 FID
    fid_score = calculate_fid(
        gen_dir='generated_samples/fid_samples',       # 生成样本目录
        real_data_dir='./data/cifar10',               # 真实数据集目录
        num_samples=args.samples,                            # 生成样本数量
        img_size=config.img_size,                     # 与模型输入尺寸一致
        batch_size=config.batch_size,                 # 使用 Config 中的 batch_size
        device='cuda' if torch.cuda.is_available() else 'cpu' 
    )
    print(f"\n最终 FID 分数: {fid_score:.4f}")
    
    # 清理生成目录
    print("\n开始清理生成目录...")
    safe_rmtree('generated_samples')
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

def load_inception_model(device='cuda'):
    """
    加载预训练的 Inception v3 模型（仅保留特征提取层）
    :param device: 计算设备（cuda/cpu）
    :return: Inception v3 模型（输出 2048 维特征）
    """
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    # 移除最后全连接层（输出 1000 类），保留到 avgpool 层（输出 2048 维特征）
    model.fc = torch.nn.Identity()  # 原 fc 层是 Linear(in_features=2048, out_features=1000)
    return model.to(device)

def extract_features(model, data_loader, device):
    """
    从数据加载器中提取 Inception v3 特征（修复后）
    :param model: Inception v3 模型（输出 2048 维特征）
    :param data_loader: 数据加载器（输出 [B, C, H, W] Tensor，范围 [-1,1]）
    :param device: 计算设备（cuda/cpu）
    :return: 特征数组（形状 [N, 2048]）
    """
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取特征"):
            # 确保批次是 Tensor（避免列表输入）
            if isinstance(batch, list):  # 真实数据：(图像张量, 标签张量)
                images, _ = batch  # 提取图像部分（形状 [B, C, H, W]）
                batch = images
            if not isinstance(batch, torch.Tensor):
                raise TypeError(f"批次数据应为 Tensor，实际类型: {type(batch)}")
            
            # 检查张量维度（必须为 4D: [B, C, H, W]）
            if batch.dim() != 4:
                raise ValueError(f"输入张量维度错误：期望 4D，实际 {batch.dim()}D")
            
            batch = batch.to(device)  # 数据已为 [-1,1] 的 Tensor，形状 [B, C, H, W]
            
            # -------------------- 预处理：调整为 Inception 输入要求 --------------------
            # 步骤1：[-1,1] → [0,1]（与 Inception 输入兼容）
            batch_0_1 = (batch + 1) / 2.0  # 形状 [B, C, H, W], [0,1]
            
            # 步骤2：调整大小到 299x299（双线性插值，保持宽高比）
            batch_resized = torch.nn.functional.interpolate(
                batch_0_1,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )  # 形状 [B, C, 299, 299]
            
            # 步骤3：中心裁剪到 299x299（确保尺寸严格匹配）
            # 注意：interpolate 后尺寸可能略大于 299，需先裁剪再缩放（避免边缘失真）
            batch_cropped = transforms.CenterCrop((299, 299))(batch_resized)  # 形状 [B, C, 299, 299]
            
            # 步骤4：归一化（使用 ImageNet 官方均值和标准差）
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            batch_normalized = (batch_cropped - mean) / std  # 形状 [B, C, 299, 299]
            
            # -------------------- 提取 Inception 特征 --------------------
            # 输入 Tensor 范围 [0,1]，尺寸 299x299，符合 Inception 要求
            batch_features = model(batch_normalized)  # 输出 [B, 2048]
            features.append(batch_features.cpu())  # 保存到 CPU 避免内存溢出
    
    # 合并所有特征为一个数组（形状 [N, 2048]）
    return torch.cat(features, dim=0).numpy()

def load_generated_samples(gen_dir, num_samples, img_size, device):
    """
    从 PNG 目录加载生成样本并预处理为模型输入格式
    :param gen_dir: 生成样本目录（PNG 格式）
    :param num_samples: 需要加载的样本数
    :param img_size: 图片尺寸（与模型输入一致）
    :param device: 计算设备（cuda/cpu）
    :return: 数据加载器（DataLoader）
    """
    # 筛选并排序 PNG 文件
    gen_files = sorted(glob.glob(f"{gen_dir}/*.png"))
    if len(gen_files) < num_samples:
        raise ValueError(f"生成样本不足！需要 {num_samples} 张，实际找到 {len(gen_files)} 张")
    gen_files = gen_files[:num_samples]  # 截断到目标数量

    # 定义生成数据的预处理（PNG → [-1,1] 张量）
    def gen_transform(img_pil):
        # 1. 转换为 [0,255] 的 NumPy 数组
        img_np = np.array(img_pil).astype(np.float32)
        # 2. 转换为 [-1,1] 范围
        img_np = (img_np / 255.0) * 2.0 - 1.0
        # 3. 转换为 [C, H, W] 张量（通道优先）
        return torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32)

    # 创建自定义数据集
    class GeneratedDataset(torch.utils.data.Dataset):
        def __init__(self, file_list, transform):
            self.file_list = file_list
            self.transform = transform

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            img_pil = Image.open(self.file_list[idx]).convert('RGB')  # 确保 RGB 格式
            img_tensor = self.transform(img_pil)
            return img_tensor

    # 创建数据加载器（确保批次维度正确）
    gen_dataset = GeneratedDataset(gen_files, gen_transform)
    gen_loader = torch.utils.data.DataLoader(
        gen_dataset,
        batch_size=config.batch_size,  # 与训练时的 batch_size 一致
        shuffle=False,                 # 特征提取无需打乱顺序
        num_workers=4,                 # 数据加载线程数（根据 CPU 核心数调整）
        pin_memory=True                # 加速 GPU 内存拷贝（仅当 device='cuda' 时有效）
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
    计算生成样本与真实数据的 FID 分数（优化后）
    :param gen_dir: 生成样本目录（PNG 格式）
    :param real_data_dir: 真实数据集目录（如 CIFAR-10 测试集）
    :param num_samples: 计算使用的样本数（需与生成样本数一致）
    :param img_size: 图片尺寸（与模型输入一致）
    :param batch_size: 批次大小（受限于 GPU 内存）
    :param device: 计算设备（cuda/cpu）
    :return: FID 分数
    """
    # -------------------- 初始化 Inception 模型 --------------------
    model = load_inception_model(device=device)

    # -------------------- 加载真实数据 --------------------
    print("加载真实数据集...")
    real_transform = transforms.Compose([
        transforms.Resize(img_size),          # 调整到模型输入尺寸（如 32x32）
        transforms.ToTensor(),                # [0,1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],             # 转换为 [-1,1]（与生成数据一致）
            std=[0.5, 0.5, 0.5]
        )
    ])
    real_dataset = CIFAR10(
        root=real_data_dir,
        train=False,  # 使用测试集
        download=True,
        transform=real_transform
    )
    # 截断到目标样本数（避免数据集过大）
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
    try:
        # 计算均值和协方差矩阵
        mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

        # 计算协方差矩阵平方根（处理数值稳定性）
        cov_product = sigma_gen @ sigma_real
        try:
            cov_sqrt = sqrtm(cov_product)
        except np.linalg.LinAlgError:
            # 数值不稳定时添加小量（对角线填充）
            cov_sqrt = sqrtm(cov_product + 1e-6 * np.eye(sigma_gen.shape[0]))

        # 处理复数情况（理论上不会出现，但数值误差可能导致）
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real

        # 计算 FID 分数
        diff = mu_gen - mu_real
        fid_score = np.dot(diff, diff) + np.trace(sigma_gen + sigma_real - 2 * cov_sqrt)
        return float(fid_score)
    except Exception as e:
        raise RuntimeError(f"FID 计算失败: {str(e)}")

def safe_rmtree(directory, verbose=True):
    """
    安全删除目录及其所有内容（递归删除）
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

# -------------------- 示例用法（可选） --------------------
if __name__ == "__main__":
    # 创建生成样本目录（如果不存在）
    os.makedirs('generated_samples/fid_samples', exist_ok=True)
    
    # 初始化配置（根据实际情况调整）
    config = Config()
    config.batch_size = 256  # 建议根据 GPU 内存调整（如 16GB 显存用 256，8GB 用 128）
    config.img_size = 32     # 与模型输入尺寸一致（如 CIFAR-10 是 32x32）
    
    # 初始化 DDIM 模型（假设已训练完成）
    ddim = DDIM(config)
    
    # # （可选）生成样本（如果需要测试）
    # ddim.sample_fid(total_n_samples=5000, batch_size=config.batch_size)
    
    # 计算 FID（示例参数）
    try:
        fid_score = calculate_fid(
            gen_dir='generated_samples/fid_samples',       # 生成样本目录（由 sample_fid 生成）
            real_data_dir='./data/cifar10',               # 真实数据集目录（CIFAR-10 测试集）
            num_samples=5000,                            # 生成样本数量（需与实际生成数量一致）
            img_size=config.img_size,                     # 与模型输入尺寸一致
            batch_size=config.batch_size,                 # 使用 Config 中的 batch_size
            device='cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
        )
        print(f"\n最终 FID 分数: {fid_score:.4f}")
    except Exception as e:
        print(f"\nFID 计算失败: {str(e)}")
    
    # # 清理生成目录（仅当计算成功时执行）
    # print("\n开始清理生成目录...")
    # safe_rmtree('generated_samples')
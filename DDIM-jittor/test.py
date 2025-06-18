import os
import glob
import numpy as np
from scipy.linalg import sqrtm
import shutil
import jittor as jt
from jittor.models import inception_v3
from jittor.transform import Compose, Resize, ToTensor, ImageNormalize
from jittor.dataset import Dataset
from jittor.dataset import DataLoader
from jittor.dataset.cifar import CIFAR10
from Config import Config
from DDIM import DDIM
from PIL import Image
from tqdm import tqdm
import logging
from typing import Tuple, List
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneratedDataset(Dataset):
    """
    生成图像数据集类，正确继承Jittor的Dataset
    """
    def __init__(self, file_list: List[str], transform_func):
        super().__init__()
        self.file_list = file_list
        self.transform_func = transform_func
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_pil = Image.open(self.file_list[idx]).convert('RGB')
        return self.transform_func(img_pil)

class FIDCalculator:
    """FID分数计算器类"""
    
    def __init__(self, batch_size: int = 256, num_workers: int = 4):
        """
        初始化FID计算器
        :param batch_size: 批次大小
        :param num_workers: 数据加载器工作进程数
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.inception_model = None
        self._load_inception_model()
    
    def _load_inception_model(self):
        """加载预训练的Inception v3模型"""
        try:
            logger.info("加载Inception v3模型...")
            self.inception_model = inception_v3(pretrained=True)
            # 移除最后的分类层，只保留特征提取部分
            self.inception_model.fc = jt.nn.Identity()
            self.inception_model.eval()
            logger.info("Inception v3模型加载成功")
        except Exception as e:
            logger.error(f"Inception v3模型加载失败: {e}")
            raise
    
    def _preprocess_images(self, images: jt.Var) -> jt.Var:
        """
        预处理图像用于Inception v3
        :param images: 输入图像 [B, C, H, W]，范围[-1, 1]
        :return: 预处理后的图像 [B, C, 299, 299]
        """
        # 范围转换 [-1, 1] -> [0, 1]
        images = (images + 1.0) / 2.0
        
        # 调整大小到299x299
        images = jt.nn.interpolate(
            images, size=(299, 299), mode='bilinear', align_corners=False
        )
        
        # ImageNet归一化
        mean = jt.array([0.5, 0.5, 0.5], dtype=jt.float32).view(1, 3, 1, 1)
        std = jt.array([0.5, 0.5, 0.5], dtype=jt.float32).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        return images
    
    def extract_features(self, data_loader: DataLoader, desc: str = "提取特征") -> np.ndarray:
        """
        从数据加载器提取Inception v3特征
        :param data_loader: 数据加载器
        :param desc: 进度条描述
        :return: 特征数组 [N, 2048]
        """
        self.inception_model.eval()
        features = []
        
        with jt.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                # 处理批次数据
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # 维度检查
                if images.ndim != 4:
                    raise ValueError(f"输入维度错误：期望4D，实际{images.ndim}D")
                
                # 预处理
                images = self._preprocess_images(images)
                
                # 特征提取
                batch_features = self.inception_model(images)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_fid_score(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """
        计算FID分数
        :param real_features: 真实数据特征 [N, 2048]
        :param gen_features: 生成数据特征 [N, 2048]
        :return: FID分数
        """
        # 计算均值和协方差
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # 处理数值稳定性
        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_gen += np.eye(sigma_gen.shape[0]) * eps
        
        # 计算协方差矩阵平方根
        cov_product = sigma_gen @ sigma_real
        cov_sqrt = sqrtm(cov_product)
        
        # 处理复数情况
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        
        # 计算FID分数
        diff = mu_gen - mu_real
        fid_score = np.dot(diff, diff) + np.trace(sigma_gen + sigma_real - 2 * cov_sqrt)
        
        return float(fid_score)
    
    def load_real_data(self, data_dir: str, num_samples: int, img_size: int = 32) -> Tuple[DataLoader, int]:
        """
        加载真实数据
        :param data_dir: 数据目录
        :param num_samples: 样本数量
        :param img_size: 图像尺寸
        :return: (数据加载器, 实际样本数)
        """
        logger.info(f"加载真实数据，目标样本数: {num_samples}")
        
        transform = Compose([
            Resize(img_size),
            ToTensor(),
            ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # 如果数据集大小小于目标样本数，使用全部数据
        if len(dataset) < num_samples:
            logger.warning(f"真实数据集大小({len(dataset)})小于目标样本数({num_samples})")
            num_samples = len(dataset)
        
        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        
        logger.info(f"真实数据加载完成，实际样本数: {len(dataset)}")
        return loader, num_samples
    
    def load_generated_data(self, gen_dir: str, num_samples: int) -> DataLoader:
        """
        加载生成的图像数据
        :param gen_dir: 生成图像目录
        :param num_samples: 样本数量
        :return: 数据加载器
        """
        logger.info(f"加载生成数据，目标样本数: {num_samples}")
        
        # 获取PNG文件列表
        gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
        if len(gen_files) < num_samples:
            raise ValueError(f"生成样本不足！需要{num_samples}张，实际{len(gen_files)}张")
        
        gen_files = gen_files[:num_samples]
        
        # 定义预处理函数
        def transform_func(img_pil):
            img_np = np.array(img_pil).astype(np.float32)
            img_np = (img_np / 255.0) * 2.0 - 1.0  # [0, 255] -> [-1, 1]
            return jt.array(img_np.transpose(2, 0, 1), dtype=jt.float32)
        
        # 创建数据集
        dataset = GeneratedDataset(gen_files, transform_func)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        
        logger.info(f"生成数据加载完成，实际样本数: {len(dataset)}")
        return loader
    
    def generate_samples_with_ddim(self, model_path: str, num_samples: int = 10000, 
                                  batch_size: int = 256, n_steps: int = 50, 
                                  save_dir: str = './generated_samples/fid_samples') -> str:
        """
        使用DDIM模型生成指定数量的样本
        :param model_path: 训练好的模型路径
        :param num_samples: 要生成的样本数量
        :param batch_size: 生成时的批次大小
        :param n_steps: 采样步数
        :param save_dir: 保存目录
        :return: 生成样本的目录路径
        """
        logger.info(f"开始使用DDIM生成{num_samples}张样本...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化配置和DDIM
        config = Config()
        ddim = DDIM(config)
        
        # 加载训练好的模型
        try:
            logger.info(f"加载模型: {model_path}")
            if model_path.endswith('.pkl'):
                # 加载模型权重
                state_dict = jt.load(model_path)
                ddim.diffusion.model.load_state_dict(state_dict)
                logger.info("模型权重加载成功")
            else:
                logger.warning(f"未知的模型文件格式: {model_path}")
                return save_dir
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
        
        # 设置模型为评估模式
        ddim.diffusion.model.eval()
        
        # 计算需要多少个批次
        num_batches = (num_samples + batch_size - 1) // batch_size
        actual_batch_size = min(batch_size, num_samples)
        
        logger.info(f"将生成{num_batches}个批次，每批次{actual_batch_size}张图片")
        
        # 生成样本
        sample_count = 0
        with jt.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="生成样本"):
                # 计算当前批次的实际大小
                current_batch_size = min(actual_batch_size, num_samples - sample_count)
                if current_batch_size <= 0:
                    break
                
                # 生成随机噪声
                x0 = jt.randn(current_batch_size, config.img_channels, config.img_size, config.img_size)
                
                # 使用DDIM采样
                samples = ddim.sample_wihtout_save(ddim.diffusion.model, x0, n_steps)
                
                # 保存样本
                for i in range(current_batch_size):
                    # 转换为PIL图像并保存
                    sample = samples[i].cpu().numpy()
                    # 从[-1, 1]转换到[0, 255]
                    sample = ((sample + 1) / 2 * 255).astype(np.uint8)
                    # 转换维度从CHW到HWC
                    sample = np.transpose(sample, (1, 2, 0))
                    
                    # 创建PIL图像
                    img = Image.fromarray(sample)
                    
                    # 保存图像
                    img_path = os.path.join(save_dir, f'sample_{sample_count:06d}.png')
                    img.save(img_path)
                    sample_count += 1
        
        logger.info(f"样本生成完成，共生成{sample_count}张图片，保存在: {save_dir}")
        return save_dir
    
    def compute_fid_with_generation(self, model_path: str, real_data_dir: str, 
                                  num_samples: int = 10000, img_size: int = 32,
                                  batch_size: int = 256, n_steps: int = 100,
                                  save_dir: str = './generated_samples/fid_samples') -> float:
        """
        使用DDIM模型生成样本并计算FID分数
        :param model_path: 训练好的模型路径
        :param real_data_dir: 真实数据目录
        :param num_samples: 生成和计算的样本数量
        :param img_size: 图像尺寸
        :param batch_size: 生成时的批次大小
        :param n_steps: 采样步数
        :param save_dir: 生成样本保存目录
        :return: FID分数
        """
        logger.info("开始FID计算流程...")
        
        # 第一步：使用DDIM生成样本
        gen_dir = self.generate_samples_with_ddim(
            model_path=model_path,
            num_samples=num_samples,
            batch_size=batch_size,
            n_steps=n_steps,
            save_dir=save_dir
        )
        
        # 第二步：计算FID分数
        try:
            fid_score = self.compute_fid(
                gen_dir=gen_dir,
                real_data_dir=real_data_dir,
                num_samples=num_samples,
                img_size=img_size
            )
            return fid_score
        except Exception as e:
            logger.error(f"FID计算失败: {e}")
            raise
    
    def compute_fid(self, gen_dir: str, real_data_dir: str, 
                   num_samples: int = 5000, img_size: int = 32) -> float:
        """
        计算FID分数的主函数
        :param gen_dir: 生成图像目录
        :param real_data_dir: 真实数据目录
        :param num_samples: 计算样本数
        :param img_size: 图像尺寸
        :return: FID分数
        """
        logger.info("开始计算FID分数...")
        
        # 加载真实数据
        real_loader, actual_num_samples = self.load_real_data(real_data_dir, num_samples, img_size)
        
        # 加载生成数据
        gen_loader = self.load_generated_data(gen_dir, actual_num_samples)
        
        # 提取特征
        logger.info("提取真实数据特征...")
        real_features = self.extract_features(real_loader, "提取真实数据特征")
        
        logger.info("提取生成数据特征...")
        gen_features = self.extract_features(gen_loader, "提取生成数据特征")
        
        # 确保特征数量一致
        min_samples = min(len(real_features), len(gen_features))
        real_features = real_features[:min_samples]
        gen_features = gen_features[:min_samples]
        
        logger.info(f"特征提取完成，使用样本数: {min_samples}")
        
        # 计算FID分数
        fid_score = self.calculate_fid_score(real_features, gen_features)
        
        logger.info(f"FID计算完成，分数: {fid_score:.4f}")
        return fid_score


def safe_rmtree(directory: str, verbose: bool = True) -> bool:
    """
    安全删除目录
    :param directory: 目录路径
    :param verbose: 是否打印详细信息
    :return: 是否删除成功
    """
    if not os.path.exists(directory):
        if verbose:
            logger.info(f"目录不存在: {directory}")
        return True
    
    if verbose:
        logger.info(f"删除目录: {directory}")
    
    try:
        shutil.rmtree(directory)
        if verbose:
            logger.info(f"删除成功: {directory}")
        return True
    except Exception as e:
        logger.error(f"删除目录失败: {e}")
        return False


def main():
    """主函数示例"""
    # 设置Jittor
    jt.flags.use_cuda = 1
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
    config.eta = args.eta
    n_samples = args.samples
    steps = args.steps
    
    # 初始化FID计算器
    fid_calculator = FIDCalculator(
        batch_size=config.batch_size,
        num_workers=4
    )
    ddim = DDIM(config)
    ddim.sample_fid(total_n_samples=n_samples,n_steps=steps)
    
    if os.path.exists('generated_samples/fid_samples'):
        logger.info("发现已有生成样本，直接计算FID...")
        try:
            fid_score = fid_calculator.compute_fid(
                gen_dir='generated_samples/fid_samples',
                real_data_dir='./data',
                num_samples=n_samples,
                img_size=config.img_size
            )
            print(f"\nFID分数: {fid_score:.4f}")
        except Exception as e:
            logger.error(f"FID计算失败: {e}")
        finally:
            safe_rmtree('generated_samples')
            logger.info("生成样本已清理")


if __name__ == "__main__":
    main()
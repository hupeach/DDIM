import jittor as jt
from jittor import Module
from jittor import nn
from jittor import transform
from jittor.dataset.cifar import CIFAR10
from jittor.dataset import DataLoader
import os
import numpy as np


def default_transform():
    """对数据进行预处理"""
    return transform.Compose([
        transform.RandomHorizontalFlip(),  # 水平翻转
        transform.ToTensor(),  # 转换为Tensor
        transform.ImageNormalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]     
        )
    ])


class Datasets:
    """CIFAR-10自定义数据集类"""
    def __init__(
        self,
        root: str='./data',
        train: bool = True,
        download: bool = True,
        transform = None,
        batch_size:int=16,
        shuffle:bool=True,
        num_workers:int=4,
        drop_last:bool=False
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        # 根据是否为训练集设定shuffle
        self.shuffle = self.train if shuffle is None else shuffle

        # 加载数据集文件
        self.dataset = self.load_data()

    def load_data(self):
        """加载训练集或测试集数据"""
        if self.train:
            dataset = CIFAR10(self.root,train=True,transform=default_transform())
        else:
            dataset = CIFAR10(self.root,train=False,transform=default_transform())
        return dataset

    def get_dataloader(self):
        return DataLoader(dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last)

    def __len__(self) -> int:
        """返回数据集总样本数"""
        return len(self.dataset)

if __name__ == "__main__":
    # 初始化训练集
    train_dataset = Datasets(
        root="./data",
        train=True,
        download=True,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    
    # 获取训练数据加载器
    train_loader = train_dataset.get_dataloader()
    
    # 验证数据加载（打印第一个batch信息）
    for batch in train_loader:
        images, labels = batch
        print(f"图像形状：{images.shape}（批次大小，通道，高，宽）")
        print(f"标签形状：{labels.shape}（批次大小，）")
        print(f"图像取值范围：[{images.min():.4f}, {images.max():.4f}]")
        print(f"标签示例：{labels[:5].tolist()}")
        break

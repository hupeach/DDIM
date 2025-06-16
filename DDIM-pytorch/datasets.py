import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Callable, Optional


def default_transform() -> Callable:
    """对数据进行预处理"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 水平翻转
        # 基础转换：转Tensor并标准化到[-1, 1]
        transforms.ToTensor(),
        # 归一化
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]  
        )
    ])


class Datasets:
    """
    数据集类
    """
    def __init__(
            self,
            root: str = "./data",
            train: bool = True,
            download: bool = True,
            transform: Optional[Callable] = None,
            batch_size: int = 128,
            shuffle: Optional[bool] = None,
            num_workers: int = 16,
            drop_last: bool = False,
    ):
        """
        初始化
        :param root: 数据集路径
        :param train: 是否选择训练集
        :param download: 是否自动下载
        :param transform: 数据集变换函数
        :param batch_size: 一次加载数据大小
        :param shuffle: 是否打乱
        :param num_workers: 多少个线程读取
        :param drop_last: 不足一个batch_size的数据是否丢弃
        """
        self.root = root
        self.train = train
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        # 根据是否为训练集设定shuffle
        self.shuffle = self.train if shuffle is None else shuffle
        # 初始化数据预处理流程
        self.transform = default_transform() if transform is None else transform
        # 加载CIFAR-10数据集
        self.dataset = self.load_dataset()

    def load_dataset(self) -> Dataset:
        """加载CIFAR-10数据集"""
        return datasets.CIFAR10(
            root=self.root,
            train=self.train,
            download=self.download,
            transform=self.transform
        )

    def get_dataloader(self) -> DataLoader:
        """获取配置好的DataLoader实例"""
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def __len__(self) -> int:
        """支持len()操作，返回数据集样本数"""
        return len(self.dataset)


# if __name__ == "__main__":
#     # 初始化训练集数据集（使用默认配置）
#     train_dataset = Datasets(
#         root="./data",
#         train=True,
#         batch_size=16,
#         num_workers=1,
#         shuffle=True
#     )
#
#     # 初始化测试集数据集（自动关闭shuffle）
#     test_dataset = Datasets(
#         root="./data",
#         train=False,
#         batch_size=16,
#         num_workers=1,
#         shuffle=False  # 显式关闭（可选，默认已处理）
#     )
#
#     # 获取DataLoader
#     train_loader = train_dataset.get_dataloader()
#     test_loader = test_dataset.get_dataloader()
#
#     # 打印数据集信息
#     print("训练集信息:")
#     print(f"  数据集路径: {train_dataset.root}")
#     print(f"  样本数量: {len(train_dataset)}")
#     print(f"  单批次形状: {next(iter(train_loader))[0].shape}")  # 输出: torch.Size([256, 3, 32, 32])
#
#     print("\n测试集信息:")
#     print(f"  样本数量: {len(test_dataset)}")
#     print(f"  单批次形状: {next(iter(test_loader))[0].shape}")
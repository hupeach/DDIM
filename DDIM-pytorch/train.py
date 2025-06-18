from DDIM import DDIM
from Config import Config
import argparse

def train():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练过程')
    parser.add_argument('-T', type=int, default=1000, 
                        help='扩散过程的总时间步（默认：1000）')
    parser.add_argument('-BS', type=int, default=256, 
                        help='采样批次大小（默认：256）')
    parser.add_argument('-epoch', type=int, default=20, 
                        help='训练轮数（默认：200）')
    parser.add_argument('-lr', type=float, default=5e-5, 
                        help='学习率（默认：5e-5）')
    args = parser.parse_args()
    config = Config()
    config.timesteps = args.T
    config.batch_size = args.BS
    config.epoch = args.epoch
    config.lr = args.lr
    ddim = DDIM(config)
    ddim.train_f()

if __name__ == '__main__':
    train()  
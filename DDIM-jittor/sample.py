from Config import Config
from DDIM import DDIM
import jittor as jt
import argparse

def sample():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练过程')
    parser.add_argument('-steps', type=int, default=20, 
                        help='采样步数（默认：20）')
    parser.add_argument('-eta', type=float, default=0.0, 
                        help='控制随机性的参数（默认：0.0）')
    parser.add_argument('-BS', type=int, default=256, 
                        help='采样批次大小（默认：256）')
    args = parser.parse_args()
    config = Config()
    config.batch_size = args.BS
    config.eta = args.eta
    ddim = DDIM(config)
    ddim.sample(n_steps=args.steps)

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    sample() 
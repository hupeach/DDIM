from DDIM import DDIM
from Config import Config
import jittor as jt


def train():
    config = Config()
    ddim = DDIM(config)
    ddim.train_f()

if __name__ == '__main__':
    jt.flags.use_cuda=1
    train()  
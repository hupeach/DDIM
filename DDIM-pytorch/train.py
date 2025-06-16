from DDIM import DDIM
from Config import Config

def train():
    config = Config()
    ddim = DDIM(config)
    ddim.train_f()

if __name__ == '__main__':
    train()  
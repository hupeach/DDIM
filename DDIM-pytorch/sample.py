from Config import Config
from DDIM import DDIM

def sample():
    config = Config()
    config.batch_size = 64
    ddim = DDIM(config)
    ddim.sample(n_steps=100)

if __name__ == '__main__':
    sample() 

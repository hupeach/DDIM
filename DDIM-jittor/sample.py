from Config import Config
from DDIM import DDIM
import jittor as jt

def sample():
    config = Config()
    ddim = DDIM(config)
    ddim.sample(n_steps=1000)

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    sample() 

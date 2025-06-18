import jittor as jt
from jittor import Module
from jittor import nn
import os
import matplotlib.pyplot as plt
from diffusion import Diffusion
from utils import get_beta_schedule

class DDIM(Module):
    def __init__(self,config):
        super(Module,self).__init__()
        self.train_loader = config.train_loader
        self.val_loader = config.val_loader
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.model = config.model
        self.n_steps = config.n_steps
        self.save_dir = config.save_dir
        self.beta = get_beta_schedule(config.beta_mode,config.n_steps)
        self.diffusion = Diffusion(config.model,config.n_steps,config.batch_size,self.beta,config.eta)
        self.optimizer = config.optimizer
        self.best_val_loss = 10

    def train(self):
        self.model.train()
        train_loss_list = []
        global_step = 0
        for epoch in range(self.epoch):
            epoch_train_loss = 0.0
            num_batches = 0
            for batch_id,(x0,_) in enumerate(self.train_loader):
                x0 = x0.transpose(0,3,1,2)
                self.optimizer.zero_grad()
                loss = self.diffusion.loss(x0)
                self.optimizer.step(loss)
                train_loss_list.append(loss)
                epoch_train_loss += loss.item()
                num_batches += 1
                global_step += 1
        # plt.plot(range(5000),train_loss_list)
        # plt.show()
            avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0
            train_loss_list.append(avg_train_loss)
            print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")
            val_loss = self.validate()
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(self.save_dir, 'best_model.pkl')
                jt.save(self.model.state_dict(), save_path)
                print(f"Saved best model at epoch {epoch} (Val Loss: {val_loss:.4f}) -> {save_path}")


    def validate(self):
        """验证集评估"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        with jt.no_grad():  # 关闭梯度计算
            for batch_id, (x0, _) in enumerate(self.val_loader):
                x0 = x0.transpose(0, 3, 1, 2)  # 维度
                loss = self.diffusion.loss(x0)  # 计算损失
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        return avg_val_loss
    

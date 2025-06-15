import copy
import jittor as jt
import tqdm
import time
import matplotlib.pyplot as plt
import logging
from logging import FileHandler
import datetime
from utils import get_time_schedule
from utils import visualize
import math
import os
import csv


class DDIM:
    def __init__(self,config):
        # 外部传入的参数
        self.timesteps = config.timesteps
        self.batch_size = config.batch_size
        self.diffusion = config.diffusion
        self.epoch = config.epoch
        self.train_dataset = config.train_dataset
        self.test_dataset = config.test_dataset
        self.train_len = math.ceil(len(config.train_dataset) / self.batch_size)
        self.test_len = math.ceil(len(config.test_dataset) / self.batch_size)
        self.train_dataloader = config.train_dataset.get_dataloader()
        self.test_loader = config.test_dataset.get_dataloader()
        self.optimizer = config.optimizer
        self.img_channels = config.img_channels
        self.img_size = config.img_size
        # 模型训练过程中需要的参数
        self.epoch_time = []
        self.epoch_train_loss = []
        self.epoch_test_loss = []
        self.min_train_loss = float('inf')
        self.min_test_loss = float('inf')
        self.min_avg_loss = float('inf')
        self.time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join('logger', f'training_metrics_{self.time}.csv')
        self.csv_written = False
        # 日志记录需要的参数
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                # 带时间戳的日志文件防止覆盖
                FileHandler(f'logger/logger_{self.time}.log'),
                # 输出到控制台
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


    def train_f(self):
        """
        训练函数，先对模型进行训练，将每个epoch输出保存到日志，将最后结果绘图保存到文件
        :return: 无
        """
        epoch_data = []
        for epoch in tqdm.tqdm(range(self.epoch),desc='训练轮次'):
            self.diffusion.model.train()
            train_loss = jt.float32(0)
            progress = tqdm.tqdm(self.train_dataloader,desc=f'第{epoch}个epoch')
            start = time.time()
            for batch_id,(data,_) in enumerate(progress):
                bloss = self.diffusion.loss(data)
                self.optimizer.step(bloss)
                batch_loss = bloss.item()
                train_loss += batch_loss
                # print(f"第{epoch}个epoch 第{batch_id}个batch的loss : {batch_loss}")
                
            epoch_time =time.time()-start
            
            self.epoch_time.append(epoch_time)
            avg_train_loss = train_loss / self.train_len
            self.epoch_train_loss.append(avg_train_loss)
            epoch_data.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_time': epoch_time
            })
            # print(f"第{epoch}个epoch的运行时间 ： {epoch_time}")
            # print(f"第{epoch}个epoch的平均损失 : {avg_loss}")
            # 对模型泛化能力进行验证
            avg_test_loss = self.eval_f()
            self.logger.info(
                f"Epoch {epoch} 完成 | "
                f"训练损失: {avg_train_loss:.4f} | "
                f"测试损失: {avg_test_loss:.4f} | "
                f"单轮耗时: {epoch_time:.2f}s"
            )
            self.save_best_models(avg_train_loss, avg_test_loss)
        self._write_csv(epoch_data)
        self.plot_curves()


    def eval_f(self):
        """
        测试函数，在测试集测试模型的泛化能力
        :return: 测试集的平均损失
        """
        self.diffusion.model.eval()
        total_test_loss = 0.0
        with jt.no_grad():
            for data, _ in self.test_loader:
                loss = self.diffusion.loss(data)
                batch_loss = loss.item()
                total_test_loss += batch_loss
                self.epoch_test_loss.append(total_test_loss)
        avg_test_loss = total_test_loss / self.test_len
        return avg_test_loss
    
    def _write_csv(self, epoch_data):
        """辅助函数：将所有epoch数据写入CSV文件"""
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_time'])
                # 写入表头
                writer.writeheader()
                # 写入所有epoch数据（保留4位小数）
                for data in epoch_data:
                    writer.writerow({
                        'epoch': data['epoch'],
                        'train_loss': round(data['train_loss'], 4),
                        'train_time': round(data['train_time'], 2)
                    })
            self.logger.info(f"训练指标已保存至：{self.csv_path}")
        except Exception as e:
            self.logger.error(f"写入CSV文件失败，原因：{str(e)}")

    def save_best_models(self,train_loss, test_loss):
        """
        由于只有一个GPU，做不了EMA，就保存三个fold的模型，后期做集成
        :param train_loss: 训练损失
        :param test_loss: 测试损失
        :return: 无
        """
        # 训练损失最小的模型
        if train_loss < self.min_train_loss:
            self.min_train_loss = train_loss
            jt.save(self.diffusion.model.state_dict(), f'model/best_train_model.pkl')
            self.logger.info(f"训练损失更优（{train_loss:.4f}），已保存模型至 model/best_train_model.pkl")

        # 测试损失最小的模型
        if test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            jt.save(self.diffusion.model.state_dict(), f'model/best_test_model.pkl')
            self.logger.info(f"测试损失更优（{test_loss:.4f}），已保存模型至 model/best_test_model.pkl")

        # 平均损失最小的模型
        avg_loss = (train_loss + test_loss) / 2
        if avg_loss < self.min_avg_loss:
            self.min_avg_loss = avg_loss
            jt.save(self.diffusion.model.state_dict(), f'model/best_avg_model.pkl')
            self.logger.info(f"平均损失更优（{avg_loss:.4f}），已保存模型至 model/best_avg_model.pkl")

    def plot_curves(self):
        """
        将训练过程的损失和时间保存成折线图
        :return: 无
        """
        plt.rcParams['font.family'] = 'SimHei'  # 设置默认字体为中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题
        # 确保保存路径存在
        save_dir = os.path.join(os.getcwd(), 'picture')
        os.makedirs(save_dir, exist_ok=True)
        # 绘制训练损失曲线
        plt.figure(figsize=(6, 4))
        plt.plot(range(self.epoch), self.epoch_train_loss, label='训练损失', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失')
        plt.legend()
        plt.tight_layout()  # 调整布局避免标签截断
        save_path = os.path.join(save_dir, f'loss_curves_{self.time}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        # 绘制训练时间曲线
        plt.figure(figsize=(6, 4))
        plt.plot(range(self.epoch), self.epoch_time, label='单轮训练时间（秒）', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('单轮训练时间（秒）')
        plt.title('单轮训练时间变化')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch_time_{self.time}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"训练曲线已保存至 {save_dir} 文件夹")


    def sample(self,x0:jt.array=None,n_steps:int=50):
        """
        采样函数，即图像生成过程
        :param x0: 初始的图片
        :param n_steps: 需要多少采样步
        :return: 采样生成的图片
        """
        self.diffusion.model.eval()
        # 加载模型
        models = self.load_models()
        # 获取时间表
        t_schedule = get_time_schedule('linear',self.timesteps,n_steps+1)
        # 集成模型
        with jt.no_grad():
            self.logger.info(f"{n_steps}步采样{self.batch_size}张图片")
            result_sample = []
            if x0 is None:
                x0 = jt.randn((self.batch_size,self.img_channels,self.img_size,self.img_size))
            t_prev = jt.tensor([self.timesteps-1],dtype=jt.long)
            start = time.time()
            for t in t_schedule:
                t = t.unsqueeze(0) if t.ndim == 0 else t
                x0_preds = []
                if t == [self.timesteps-1]:
                    continue
                else:
                    # for model in models:
                    #     x0_pred = model.p_sample(x0,t,t_prev)
                    #     x0_preds.append(x0_pred)
                    # x0 = torch.mean(torch.stack(x0_preds), dim=0)
                    model = models[2]
                    x0 = model.p_sample(x0,t_prev,t)
                    t_prev = t
                    result_sample.append(x0)
            cost = time.time()-start
            self.logger.info(f"本次采样{self.batch_size}张图片花费{cost}s")
        # 将生成的图片保存
        os.makedirs('output', exist_ok=True)
        final_samples = result_sample[-1]  # shape:(B,C,H,W)
        save_img_path = os.path.join('output', f'samples_result_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        visualize(final_samples,self.batch_size,save_img_path,dpi=1000)
        self.logger.info(f"生成样本已保存至：{save_img_path}")

    def load_models(self):
        """
        加载模型的函数
        :return: 返回模型的列表
        """
        model_path = [
            'model/best_avg_model.pkl',
            'model/best_test_model.pkl',
            'model/best_train_model.pkl'
        ]
        models = []
        for path in model_path:
            new_diffusion = copy.deepcopy(self.diffusion)
            try:
                new_diffusion.model.load_state_dict(jt.load(path))
                new_diffusion.model.eval()
                models.append(new_diffusion)
                self.logger.info(f"加载模型{path}成功")
            except Exception as e:
                self.logger.info(f"加载模型{path}失败 原因：{e}")
                raise
        return models
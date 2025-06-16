import os
import torch
import logging
from tqdm import tqdm
from Config import Config
from diffusion import DenoiseDiffusion
from DDIM import DDIM
from datasets import Datasets
from utils import visualize, get_time_schedule

def test_ddim_model():
    # --------------------------
    # 环境准备与参数设置
    # --------------------------
    # 设置随机种子保证可复现性
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"当前使用设备: {device}")
    if device == 'cpu':
        print("警告：未检测到GPU")

    # --------------------------
    # 配置初始化（简化参数）
    # --------------------------
    config = Config()
    # 调整关键参数加速测试（根据硬件调整）
    config.batch_size = 16  # 减小批量大小适应内存
    config.epoch = 5        # 仅训练5轮验证流程
    config.img_size = 32    # 保持32x32小尺寸加速计算
    config.channels = 32    # 减少特征子空间数目
    config.timesteps = 100  # 减少时间步加速训练
    config.eta = 0.0        # DDIM确定性模式

    # --------------------------
    # 数据集加载验证
    # --------------------------
    try:
        # 初始化数据集（使用CIFAR-10）
        train_dataset = Datasets(root='./data', train=True, batch_size=config.batch_size)
        test_dataset = Datasets(root='./data', train=False, batch_size=config.batch_size)
        
        # 验证数据加载
        train_loader = train_dataset.get_dataloader()
        test_loader = test_dataset.get_dataloader()
        
        # 检查数据形状
        sample_data, _ = next(iter(train_loader))
        assert sample_data.shape == (config.batch_size, 3, 32, 32), \
            f"数据形状错误！期望 (16,3,32,32)，实际 {sample_data.shape}"
        print("数据集加载与形状验证通过")

    except Exception as e:
        print(f"数据集加载失败: {str(e)}")
        return

    # --------------------------
    # 模型初始化验证
    # --------------------------
    try:
        # 初始化扩散模型
        diffusion = DenoiseDiffusion(
            model=config.model,
            timesteps=config.timesteps,
            eta=config.eta,
            device=device
        )
        
        # 验证模型参数设备
        assert next(diffusion.model.parameters()).device == device, \
            "模型参数未正确移动到目标设备"
        
        # 初始化DDIM训练器
        ddimer = DDIM(config=config)
        print("模型初始化与设备验证通过")

    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return

    # --------------------------
    # 模型训练验证
    # --------------------------
    try:
        print("\n===== 开始模型训练 =====")
        ddimer.train_f()  # 执行训练
        
        # 验证训练损失是否下降（简单检查）
        if len(ddimer.epoch_train_loss) == 0:
            raise RuntimeError("训练过程中未记录损失值")
        
        initial_loss = ddimer.epoch_train_loss[0]
        final_loss = ddimer.epoch_train_loss[-1]
        assert final_loss < initial_loss * 0.9, \
            f"训练损失未显著下降（初始: {initial_loss:.4f}, 最终: {final_loss:.4f}）"
        print(f"训练完成，损失从 {initial_loss:.4f} 降至 {final_loss:.4f}")

    except Exception as e:
        print(f"❌ 训练过程失败: {str(e)}")
        return

    # --------------------------
    # 模型采样验证
    # --------------------------
    try:
        print("\n===== 开始生成样本 =====")
        # 加载最佳模型
        ddimer.diffusion.model.eval()
        
        # 生成样本
        with torch.no_grad():
            # 生成随机初始噪声
            x0 = torch.randn(
                (config.batch_size, config.img_channels, config.img_size, config.img_size),
                device=device
            )
            
            # 采样参数）
            n_steps = 50  # 测试使用较少步数
            t_schedule = get_time_schedule(
                name='linear',
                timesteps=config.timesteps,
                n_steps=n_steps,
                device=device
            )
            
            # 执行采样
            result_sample = []
            t_prev = torch.tensor([config.timesteps-1], dtype=torch.long, device=device)
            for t in t_schedule:
                t_tensor = t.unsqueeze(0).to(device)
                if t_tensor == config.timesteps-1:
                    continue
                # 使用最后一个模型
                x0 = ddimer.diffusion.model.p_sample(x0, t_tensor, t_prev)
                t_prev = t_tensor
                result_sample.append(x0)
            
            # 取最后一步的样本
            generated_samples = result_sample[-1]
            
            # 验证生成样本形状
            assert generated_samples.shape == (config.batch_size, 3, 32, 32), \
                f"生成样本形状错误！期望 (16,3,32,32)，实际 {generated_samples.shape}"
            
            # 保存生成样本（范围[-1,1]转换为[0,255]）
            save_path = 'output/test_samples.png'
            visualize(generated_samples, config.batch_size, save_path, dpi=500)
            print(f"生成样本已保存至 {save_path}")

    except Exception as e:
        print(f"采样过程失败: {str(e)}")
        return

    # --------------------------
    # 清理与总结
    # --------------------------
    print("\n===== 测试总结 =====")
    print("所有关键环节验证通过！模型训练和采样流程正常工作。")

if __name__ == "__main__":
    # 清理旧日志
    if os.path.exists('logger'):
        import shutil
        shutil.rmtree('logger')
    
    # 运行测试
    test_ddim_model()
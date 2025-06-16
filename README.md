# 项目介绍
本项目仓库在Jttor和Pytorch框架下实现了DDIM论文复现，并在Cifar10数据集（使用Jittor和Pytorch封装好的导入）上对二者进行实验，本项目仓库主要包括以下内容：

 

 - [x] 环境搭建（Jittor and Pytorch）
 - [x] 模型架构（U-Net优化）
 - [x] 实验（对比Jittor和Pytorch两种框架下DDIM的效果）
 - [x] 总结（对Jittor的 ~~踩坑~~ 使用观感以及优势总结）
 - [x] 附录（包含训练脚本、采样脚本、测试脚本）
 - [x] 参考代码仓库（参考文献，帮助我解决了很多问题）
 - [x] 随笔（记录一点胡思乱想）
 # 环境搭建
 - 机器环境：
	 - RTX 4090D(24GB) * 1
	 - 18 vCPU AMD EPYC 9754 128-Core Processor 
 - Jittor版本：
	  - Jittor 1.3.9
	  - Python 3.8(Ubuntu-20.04 )
	  - CUDA 11.3
 - Pytorch版本：
	 - PyTorch 1.10.0
	 - Python 3.8(Ubuntu-20.04)
	  - CUDA 11.3

 # 模型架构
 - U-Net
 - DDPM扩散
 - DDIM采样
 - 模型训练
 # 实验
 - 损失曲线
 - 时间曲线
 - 生成质量（FID）
 - 超参搜索
 # 总结
 - Jittor的使用观感
	 - 设备：jittor.flags.use_cuda = 1一行代码就把所有的东西都放到GPU了，虽然对显存不太友好，但是不用担心pytorch报错计算的两个变量不在同一个设备了。
	 - 类型 ：这是真踩了个大坑，float32和int64不兼容，报错会报一些奇奇怪怪的错误，总之就不会定位到这里，可能是异步的原因。
	 - 进度条:tqdm对Jittor训练过程剩余时间显示的不准确，说是10个小时一轮给我吓坏了，然后进度条突飞猛进90s训完了。
 - Jittor优势：Jittor官方文档特意强调了性能测试和显存优化，所以我想从这两个方向看看Jittor的优势。
	 - 计算速度快：算的是真的快。
	 - 显存利用率高 ：可以开更大的batch_size，学习更多的特征子空间。
 # 附录
 - 训练脚本：
		 脚本包括以下参数：
		 -T：扩散过程的时间步（默认1000）
		 -BS：batch_size（默认256）
		 -epoch:训练轮数（默认150）
		 -lr:学习率（默认1e-4）
 - 采样脚本
 - 测试脚本 
 # 参考代码仓库
 - [DDPM原论文实现](https://github.com/hojonathanho/diffusion)：个人感觉DDPM的官方实现框架代码真的很清晰，我整体的训练框架也是参考了这篇代码仓库。

 - [DDIM原论文实现](https://github.com/ermongroup/ddim)：主要参考了DDIM源码在层间逻辑的改进，以及对优化器的设置，包括训练过程中的梯度裁剪，同时时间的采样参考了该源码中的对抗采样。
 - [计图官方文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)：这个文档帮助我了解了计图的语法，同时发现了一些很有意思的事情，比如计图可以手搓算子，而且在官方文档里确实有一些有意思的东西，似乎作者也在对标Pytorch，看文档可以发现Jittor对显存的利用率非常高，但也带来了一些性能损失，从我的实验中也能看到，但是我觉得这是一个不错的tradeoff，对于我这种显存不太充足的学生党来说，我完全可以开比pytorch更大的batch_size，这对我来说是一个非常不错的点。
 - [DDPM的Pytorch版本](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm)：官方代码是TensorFlow，作为一个学生党我还是更熟悉Pytorch一点。
 - [OpenAI对DDIM的开源代码](https://github.com/openai/improved-diffusion/tree/main)：这个代码就没那么容易读了，主要是参考了他的一些参数调度。
 - [苏剑林的DDPM实现](https://github.com/bojone/Keras-DDPM/tree/main)：用的Keras实现，这篇文章我主要是参照了他的损失，相对于mse损失，这篇代码采用了l2损失，并且还做了一些改动减少了计算量。
 # 随笔
 我看了很多代码和文章，我想要做出一点优化，无非要从UNet和参数调度上来做，而很多参数调度比较成熟，我要做的话其实也只能从优化问题的角度去做，但是短时间内我有很难打磨出一个比较好的优化目标。于是我想要在UNet上做出一些优化，所以我在训练模型的时候就一直在思考DDPM原来的模块作用是什么，为什么要这样做。

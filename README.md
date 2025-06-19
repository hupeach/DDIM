# 项目介绍
本项目仓库在Jttor和Pytorch框架下实现了DDIM论文复现，并在Cifar10数据集（使用Jittor和Pytorch封装好的导入）上对二者进行实验，本项目仓库主要包括以下内容：

 

 - [x] 环境搭建（Jittor and Pytorch）
 - [x] 模型架构
 - [x] 实验（对比Jittor和Pytorch两种框架下DDIM的效果）
 - [x] 总结（对Jittor的优势总结、 ~~踩坑指北~~ 使用观感）
 - [x] 附录（包含训练脚本、采样脚本、测试脚本）
 - [x] 参考代码仓库（参考文献，记录我借鉴了哪些内容）

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
	![The structure of Unet](https://github.com/hupeach/DDIM/blob/main/pictures/table-unet.png)
 - DDPM扩散：
	 - beta参数调度：设置两种参数调度，linear和cosine，
 - DDIM采样：
	 -  Respacing的Subset选择：设置三种，linear、quad、cosine。
 - 模型训练：
	 -  时间步采样：设置两种，随机时间步采样、对抗时间步采样。
	 - 损失函数：mse损失，经过个人实验验证，效果在cifar10上优于l2。
	 - 梯度裁剪：很经典的防止梯度爆炸的方法，最好要有，虽然有Adam的自适应学习率，但是梯度依旧容易爆炸。
	 - 集成学习：训练了三个模型，做了一个集成，但是最后没有采用，因为比较影响采样速度。
 # 实验
 - 损失曲线：可以看到损失相差不大
 ![Jittor损失](https://github.com/hupeach/DDIM/blob/main/pictures/jittor_loss.png)
 ![Pytorch损失](https://github.com/hupeach/DDIM/blob/main/pictures/pytorch_loss.png)
 ![损失对比](https://github.com/hupeach/DDIM/blob/main/pictures/cmp_loss.png)
 - 时间曲线：可以看到很明显的Jittor要快很多。
 ![Jittor训练时间](https://github.com/hupeach/DDIM/blob/main/pictures/jittor_time.png)
 ![Pytorch训练时间](https://github.com/hupeach/DDIM/blob/main/pictures/pytorch_time.png)
 ![训练时间对比](https://github.com/hupeach/DDIM/blob/main/pictures/cmp_time.png)
 - 生成质量（T=1000的模型在10000样本采样下的FID）：
	 - pytorch：
       |pytorch|steps=10|steps=20|steps=50|steps=100|
       |--|--|--|--|--|
       | η=0.0 | 35.1979 |26.2736|27.3667|30.3170|
       |η=0.2|35.1298|26.4229|28.0837|29.7941|
       |η=0.5|35.6484|26.9818|27.5941|29.8717|
       |η=1.0|35.3288|27.0929|28.2088|29.3678
	 - jittor：
        |jittor|steps=10|steps=20|steps=50|steps=100|
       |--|--|--|--|--|
       | η=0.0 | 35.3697 |24.3581|23.1885|24.0099|
       |η=0.2|35.7621|24.4647|23.4794|24.8441|
       |η=0.5|35.2226|24.4424|23.4703|24.4026|
       |η=1.0|35.2685|23.9264|23.3636|24.7969|
  - 采样结果展示（20步采样64张图片）
	 - eta =0.0 :
	![Jittor-step20-1.834s](https://github.com/hupeach/DDIM/blob/main/DDIM-jittor/output/eta%3D0.0.png)
  	![Pytorch-step20-0.779s](https://github.com/hupeach/DDIM/blob/main/DDIM-pytorch/output/eta%3D0.0.png)
	- eta = 0.2:
	![Jittor-step20-1.833s](https://github.com/hupeach/DDIM/blob/main/DDIM-jittor/output/eta%3D0.2.png)
	![Pytorch-step20-0.777s](https://github.com/hupeach/DDIM/blob/main/DDIM-pytorch/output/eta%3D0.2.png)
	- eta = 0.5:
	![Jittor-step20-1.782s](https://github.com/hupeach/DDIM/blob/main/DDIM-jittor/output/eta%3D0.5.png)
	![Pytorch-step20-0.784s](https://github.com/hupeach/DDIM/blob/main/DDIM-pytorch/output/eta%3D0.5.png)
	- eta = 1.0:
	![Jittor-step20-1.795s](https://github.com/hupeach/DDIM/blob/main/DDIM-jittor/output/eta%3D1.0.png)
	![Pytorch-step20-0.783s](https://github.com/hupeach/DDIM/blob/main/DDIM-pytorch/output/eta%3D1.0.png)
- 实验结果发现与反思：
	 - 发现：
		 - FID变化趋势：pytorch和jittor版本大体相同。
		 - trade off：性能和效率。 
			 - 效率：可以看到Jittor在长时间的训练过程中表现优异。
			 - 性能：在短时间的采样过程中时间较长。
			 - 原因：我猜测可能是因为Jittor全部扔到GPU计算，而将数据加载到GPU的速度较慢。
		 - 采样结果：部分图片仍存在噪声，因此模型需要进一步的训练。
		 - 推理速度：Jittor推理速度远远慢于Pytorch，猜测Jittor将需要计算的数据全部加载到GPU，所以速度较慢。
	 - 思考后续方向：
		 - 采样的时间步子集选择：或许可以构建一个关于t(1,2,...n)的多目标优化，而不是按照直觉人工设计子集。
		 - 损失函数：mse和l2产生了不同的效果，我觉得或许可以设计一个对抗损失来优化损失函数，在各个通道差异和整张图片差异之间找到一个平衡点。
	 - 为什么没有达到原论文效果：
		- 训练轮次：我训练的时间还是偏少的，大概也就6h，而且后面损失依旧偶尔会下降，依旧有可搜索的解空间。
		- 模型处理：原论文使用的EMA平滑化了模型参数， 我由于单卡训练，并没有选择EMA，而Assemble对采样速度影响较大，所以模型处理方面存在欠缺。
		- 数据集：数据量较小，仅有50000张，所以可能较大程度上影响了模型性能。
	- 为什么FID分数比较奇怪：
		- 随机性：训练轮数较少，达不到大数定律的标准，也不能够让模型随机性趋于一致。
		- 采样公式：采样公式是为了跳步设计的，而我训练采用的是DDPM，这是基于马尔科夫性设计的，如果不跳步或者跳步少的话的话，会产生较大的误差。这点可以从实验结果只有100steps时FID随着eta上升而下降可以看出来，因为eta上升代表采样公式越来越接近DDPM。
 # 总结
  - Jittor优势：Jittor官方文档特意强调了性能测试和显存优化，由于我为了对比没有按照文档方法优化显存，所以我将目标放在了性能测试和切身体验来说明。
	 - 计算速度快：算的是真的快，我调整模型都是从Jittor上调的，比从Pytorch上快不少，而且对性能影响不大。
	 - 接口便利：比如把计算迁移到GPU，优化器更新梯度等，Jittor将他们封装到一个接口，写起来方便一点。
	 
 - Jittor的使用观感
	 - 显存：Jittor对GPU的利用率达不到100%，比如我的卷积的通道数，640在Pytorch上可以正常运行，但是在Jittor上不行。
	 - 耦合度太高：Jittor的一些接口耦合度稍高，这是便利带来的后果。
	 - 类型 ：这是真踩了个大坑，float32和int64不兼容，报错会报一些奇奇怪怪的错误，然后让我提交issue，总之就不会定位到这里，可能是异步的原因。
	 - 进度条：tqdm对Jittor训练过程剩余时间显示的不准确，说是10个小时一轮给我吓坏了，然后进度条突飞猛进90s训完了。
	 - 数据集加载：与pytorch不同的是自定义数据集一定要加入self.batch_size,self.num_worker,self.shuffle这三个，要不然会报错。
	
 # 附录
 - 训练脚本：train.py文件，自动运行训练
	>python train.py -T 1000 -BS 256 -epoch 200 -lr 5e-5  
	-T：扩散过程的时间步（默认1000）
		 -BS：batch_size（默认128）
		 -epoch:训练轮数（默认200）
		 -lr:学习率（默认1e-4）
		 
 - 采样脚本：sample.py文件，自动运行采样，BS要可开放，因为生成图片在一个画布，中间有开放运算。
 
	>python sample.py -steps 20 -eta 0.5 -BS 64
	-steps：采样步数（默认20）
	-eta：控制随机性的超参数（默认0.0）
	-BS：采样的batch_size（默认64）

 - 测试脚本 ：test.py文件，自动运行FID分数测试
	>python test.py -samples 10000 -eta 0.5 -BS 256 
	-samples：采样的样本数（默认10000）
	-eta：采样的超参数（默认0.0）
    -BS：采样的batch_size（默认256） 
	
 # 参考代码仓库
 - [DDPM原论文实现](https://github.com/hojonathanho/diffusion)：个人感觉DDPM的官方实现框架代码真的很清晰，我整体的训练框架也是参考了这篇代码仓库。

 - [DDIM原论文实现](https://github.com/ermongroup/ddim)：主要参考了DDIM源码在层间逻辑的改进，以及对优化器的设置，包括训练过程中的梯度裁剪，但是DDIM得对抗采样在我这里效果比较差，理论上应该效果好。
 - [计图官方文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)：这个文档帮助我了解了计图的语法，对于pytorch和jittor的一些对齐方式也是参照文档来的，而且计图似乎可以手搓算子，这点蛮有意思的。
 - [DDPM的Pytorch版本](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm)：官方代码是TensorFlow，作为一个学生党我还是更熟悉Pytorch一点。
 - [OpenAI对DDIM的开源代码](https://github.com/openai/improved-diffusion/tree/main)：这个代码就没那么容易读了，主要是参考了他的一些参数调度。
 - [苏剑林的DDPM实现](https://github.com/bojone/Keras-DDPM/tree/main)：用的Keras实现，这篇文章我主要是参照了他的损失，相对于mse损失，这篇代码采用了l2损失，并且还做了一些改动减少了计算量，但是我的l2的效果不如mse的效果。
 

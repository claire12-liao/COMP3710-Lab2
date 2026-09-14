# Demo 讲解与定位卡

先完成实际运行，再按真实结果讲。以下英文是表达参考；遇到未完成的项目应说尚未完成，不把代码存在说成训练达标。

## 约3分钟自主陈述：按实际完成内容取用

“This project compares signal representations, traditional machine learning and deep learning.”

这个项目比较信号表示、传统机器学习和深度学习。

“In Part 1, I reconstruct a square wave from odd harmonics and recover its frequency components with the DFT. I compare a naive implementation, NumPy FFT and an explicit PyTorch tensor implementation across different input sizes.”

Part 1 用奇次谐波重建方波，再用 DFT 分解频率。展示实际计时图，按图讲最快到最慢。只有真实跑过 CUDA 才补充 GPU 对比。

“The FFT and direct DFT agree numerically. The FFT uses a more efficient algorithm, while the GPU parallelizes the direct calculation. I synchronize CUDA before and after timing.”

两种方法数值一致。FFT 改进算法复杂度，GPU 并行直接计算。CUDA 计时前后同步。

“For face recognition, I fit PCA using only the training data, then train a Random Forest on the projected features. I compare it with a two-layer CNN using the same held-out split.”

只用训练集拟合 PCA；投影后的特征送入随机森林。CNN 与 RF 使用相同测试集。指着实际 accuracy 和分类报告讲，不预先声称 CNN 更好。

“For CIFAR-10, the ResNet-18 is built from basic layers. I use data augmentation, a learning-rate schedule and mixed precision. Here are the measured test accuracy and training time.”

ResNet-18 由基础层搭建，使用增强、学习率调度、混合精度。指向真实 test accuracy 与总耗时。集群现场运行 inference 和一个完整 epoch。

“For MRI, I use separate subjects for training, validation and testing. The VAE learns a latent representation; the UNet predicts four tissue-label categories per pixel; the GAN learns to generate images through an adversarial objective.”

MRI 按受试者分集。VAE 学潜在表示，UNet 每像素四类分割，GAN 通过对抗学习生成图像。只讲自己完成并理解的任务。

“My results and checkpoints are saved, and the repository records my changes. I used AI assistance and documented what I checked and modified.”

结果和模型已保存，GitHub 记录实际修改。说明 AI 帮助及自己真实验证、改进的部分。

## 快速代码定位

VS Code：Ctrl+Shift+F 搜索以下标记；Colab：“关键代码定位”单元格可以显示对应函数。

| 标记 | 文件/函数 | 一句话解释 |
|---|---|---|
| DEMO P1-A | lab/part1.py / square_wave_fourier | 奇次频率广播，叠加重建 |
| DEMO P1-B | lab/part1.py / naive_dft_torch | 显式DFT矩阵与信号相乘，未调用FFT |
| DEMO P1-C | lab/part1.py / timed | 预热、多次中位数、CUDA同步 |
| DEMO P2-A | lab/faces.py / load_faces | 分层train/validation/test，同一split比较模型 |
| DEMO P2-B | lab/faces.py / pca_rf | 仅训练集均值和SVD，避免泄漏 |
| DEMO P3-A | lab/models.py / FaceCNN | 两层卷积，32 filters，3×3 |
| DEMO P3-B | lab/models.py / ResidualBlock.forward | F(x)+shortcut，残差相加 |
| DEMO P3-C | lab/faces.py / main | NCHW灰度输入，加通道维度 |
| DEMO P3-D | lab/cifar.py / augment_batch | GPU批量裁剪、翻转、cutout |
| DEMO P3-E | lab/cifar.py / train_epoch | AMP前向，缩放梯度，参数更新 |
| DEMO P3-F | lab/cifar.py / main | Demo真跑完整一轮，保留原最佳模型 |
| DEMO P4-A | lab/models.py / VAE.forward | mu + sigma*epsilon |
| DEMO P4-B | lab/models.py / UNet.forward | 上采样与encoder特征拼接 |
| DEMO P4-C | lab/data.py / MRIDataset | 标签最近邻缩放，四灰度映射 |
| DEMO P4-D | lab/mri.py / vae_loss | MSE + beta*KL |
| DEMO P4-E | lab/mri.py / segmentation_loss | one-hot交叉熵与soft Dice组合 |
| DEMO P4-F | lab/mri.py / dice_from_confusion | 按四类计算真正DSC |
| DEMO P4-G | lab/mri.py / visualise_vae | 高维latent的二维切片 |
| DEMO P4-H | lab/mri.py / supervised | 验证选模后，用测试集报告/现场推理 |
| DEMO P4-I | lab/mri.py / gan | detach阻断G梯度，只在训练D时用 |

## Part 1 常见问答

**What does the Fourier transform do?**

It represents a signal by its frequency components. 它把随时间变化的信号转换成各个频率的组成。

**Why only odd harmonics?**

For this symmetric 50%-duty square wave, the even harmonics cancel by symmetry. 这里是对称、50%占空比方波，对称性使偶次谐波抵消；不是所有波形都只有奇次。

**Does 50 harmonics mean 50 Hz?**

No. With a 1 Hz fundamental, the first 50 odd harmonics run from 1 to 99 Hz. 50是项数，最高频率99 Hz。

**Why are there spikes?**

The Gibbs phenomenon produces overshoot near discontinuities. More terms narrow the affected region, but the overshoot does not vanish. 更多项让跳变更陡，但尖峰不会彻底消失。

**Why does the reconstruction go through zero at the jump?**

At a discontinuity the series approaches the midpoint of the left and right limits. 对从1到-1的跳变，中点是0。

**Are DFT and FFT different transforms?**

No. FFT is a fast algorithm for computing the DFT. 它们计算的是同一个变换。

**What are the complexities?**

Direct DFT is O(N²), while FFT is typically O(N log N). GPU parallelization does not change the direct DFT's quadratic work. 不能说用了GPU就把DFT变成FFT。

**Why might CPU FFT beat GPU direct DFT?**

FFT needs much less work; small GPU jobs also have launch overhead. GPU不一定最快，尤其这里比较的算法本来不同。

**What is the timing scope?**

Warm-run medians. GPU synchronization brackets the work; the DFT matrix is built during each call; CPU-GPU data transfers are excluded. 包含矩阵生成，不含数据拷贝，需如实说明。

**Why float64/complex128?**

For a clear numerical comparison and small roundoff error. FP64 can be slow on a T4, which is a relevant limitation to explain. 这种选择偏重数值验证，不代表GPU的最高吞吐性能。

**Why require N > 198?**

The highest included frequency is 99 Hz over one second, so the sampling rate must exceed twice that frequency. 防止混叠。本实验选择256、512、1024、2048。

## Part 2 / 3.1 常见问答

**What is PCA?**

PCA finds orthogonal directions that capture the greatest variation in the training data. PCA寻找数据变化最大的正交方向。

**What is an eigenface?**

A principal direction reshaped into an image; it is a pattern of variation, not necessarily a real person's face. 主成分还原成图像形状，不是真实人脸照片。

**Why subtract the mean?**

We model variation around the training mean. We use that same mean for validation and test data. 测试集不能重新单独计算用于拟合的均值。

**What do U, S and Vh mean?**

For centered X, X = U S Vh. Rows of Vh give feature-space principal directions; S²/(n_train−1) gives their variances. 代码取Vh前150行，用X_centered @ components.T投影。

**What does compactness show?**

The cumulative fraction of variance retained as we add components. 不是分类准确率；保留更多方差不保证分类更好。

**What does Random Forest do?**

It combines many decision trees to predict the identity from the PCA features. 多棵树共同分类；PCA本身没有预测人名。

**What is the difference between the two pipelines?**

PCA learns a linear representation without labels, then RF classifies it. CNN learns spatial features and classification jointly from labels. CNN保留图片二维位置关系。

**What is NCHW?**

Batch size, channels, height and width. Gray images have one channel. 灰度人脸形状为[N,1,H,W]。

**Why no softmax before CrossEntropyLoss?**

The loss already incorporates log-softmax, so the model returns logits. 预测展示概率时才另用softmax。

**Does the CNN have to perform better?**

It is an experimental comparison; a small imbalanced dataset can favor a simpler model. 报真实结果并解释，小数据时不能保证CNN胜出。

## ResNet / 训练 常见问答

**What is a residual connection?**

It adds a shortcut to a learned residual, making optimization of deeper networks easier. 维度变化时用1×1卷积匹配shortcut形状。

**Is this a prebuilt ResNet?**

The code defines the blocks and layers directly. It uses eight two-convolution residual blocks plus the stem and classifier. 没有调用torchvision.models或加载预训练权重；CIFAR stem改成3×3、stride1。

**What is mixed precision?**

Selected operations use lower precision to reduce memory and speed up training; gradient scaling protects small gradients. 不等于把所有变量都简单转成half。

**What is an epoch?**

One pass through the training set. 这里45,000张训练图走一遍是完整epoch，不是单个batch。另5,000张为validation。

**How is the best model chosen?**

By validation accuracy. The held-out test set is evaluated after selecting the checkpoint. 不能每轮看test并据此调整超参数。

**What exactly counts toward the reported time?**

Training epochs, validation and checkpoint writes; dataset download/loading is separate. The continuous-run 94%/360-second flag is only true when the actual held-out test result and measured time meet both conditions. 时间口径与官方挑战若不同需向tutor说明，不把validation到94%当test到94%。

**Why must I use Rangpur?**

The lab explicitly requires inference and one training epoch on Rangpur during the demo. Colab GPU access does not satisfy that location requirement.

## MRI 常见问答

**Why split by subject?**

Neighboring slices from one subject are very similar. Mixing them across train and test can inflate the score. 原始数据train302人、val35人、test17人，保持原split。

**What does a VAE learn?**

An encoder estimates a latent distribution; a decoder reconstructs images from samples. VAE学mu和logvar，不只学一个确定编码。

**Why use reparameterization?**

We sample epsilon independently and compute z = mu + sigma*epsilon, so gradients can reach the encoder. 随机性放在epsilon中，参数仍可微。

**What does your VAE loss mean?**

Pixel-mean MSE plus beta times mean KL. The KL weight warms up during training, while validation uses the fixed target beta. 这是归一化、加权的重建+KL目标；不要说成未经缩放的原始ELBO数值。

**Is the manifold plot the entire latent space?**

No. It varies z0 and z1 while holding other dimensions at zero. A separate PCA scatter plot summarizes the encoded test means. 需要准确称“二维切片”。

**Why UNet skip connections?**

They recover fine spatial details from the encoder while the decoder upsamples. UNet是拼接；ResNet是相加。

**Why nearest-neighbor resizing for masks?**

Interpolating class labels would invent invalid intermediate classes. 图片可双线性，标签只能最近邻。

**How are labels represented?**

Raw values 0,85,170,255 map to class IDs 0,1,2,3. The loss explicitly constructs four-channel one-hot targets; inference saves softmax probabilities and one-hot predictions. 不凭空把未提供的类别语义命名为某组织。

**What is Dice?**

DSC = 2TP / (2TP + FP + FN). We report all four classes, not just overall accuracy or background. 完全重合为1，没有交集为0。

**What if mean Dice >0.9 but one class <0.9?**

That does not satisfy the lab's “for all labels” requirement. 逐类看，不能用平均值掩盖。

**What is GAN mode collapse?**

Different noise inputs produce nearly the same image. Check diverse noise samples and interpolation; a low loss does not prove realistic, diverse brains. 需tutor判断逼真程度。

**Why detach generated images when training D?**

That step should update the discriminator without updating the generator. During the G step the gradient must flow through D back into G. detach只在D步使用。

**Do nearest-training comparisons prove no memorization?**

No. They are a simple low-resolution diagnostic, not a proof or a realism score. 需要结合更多图片和训练过程解释。

## 当前不能背成“已经完成”的内容

除非你已经实际做到，否则不要说：我完成了Git短课；我在Rangpur现场运行过；我的CIFAR达到了94%/360秒；我的UNet每类都超过0.9；GAN已经通过逼真度检查；所有代码都是我独立写的。

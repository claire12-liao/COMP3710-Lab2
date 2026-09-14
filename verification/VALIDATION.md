# 实际验证记录 — 请不要当作完整训练成绩

验证环境：Linux CPU，Python 3.12.14，PyTorch 2.14.0+cpu，NumPy 2.3.5。没有CUDA GPU，未连接学生的Colab/Rangpur账号。验证日期：2026-09-13。

## 已实际完成

| 检查 | 数据/规模 | 结论 |
|---|---|---|
| 源码语法 | lab/、tools/ | Python编译检查通过；Slurm脚本bash语法通过 |
| Part1数值与计时 | N=256/512/1024/2048，每种方法预热后3次 | NumPy朴素DFT、NumPy FFT、PyTorch显式CPU DFT均数值一致；图和JSON已生成 |
| MRI全量审核 | 22,656个PNG | 图片与标签一一对应；四标签值0/85/170/255；各split受试者互斥 |
| LFW真实下载 | scikit-learn官方加载器 | 当前环境下载返回HTTP403，未完成真实LFW训练，不提供伪造准确率 |
| PCA/RF管线 | 合成数据180×16×12，3个合成类别 | SVD、投影、RF、分类报告/图生成通过；均值只来自训练集，PCA基正交。不是LFW结果 |
| LFW CNN结构 | 合成张量 | 恰好两层3×3/32filters卷积，输出形状与反向传播通过 |
| ResNet18 | 真实CIFAR-10的64训练/32验证/32测试小样本，2epochs | 下载、增强、训练、最佳模型保存与重载、测试和图片生成通过；准确率不具评分意义 |
| VAE | 真实MRI，各split选64张；base8，2epochs | 训练、KL/重参数化、重建、二维latent切片、PCA投影和模型重载推理通过 |
| UNet | 真实MRI，各split选32张；64×64、base8，2epochs | 训练、one-hot损失、逐类Dice、预测图片、四通道/one-hot文件输出和模型重载通过；未达到0.9 |
| GAN | 真实MRI的64张训练图，base8，2epochs | G/D更新、detach梯度检查、保存重载、生成/插值/最近训练图比较通过；图像尚不逼真 |
| 独立正确性检查 | tools/selftest.py | 6组检查通过，包括DFT预期振幅、模型结构、有限梯度、VAE参数梯度、Dice完美/不相交边界和GAN梯度隔离 |
| Colab Notebook静态与引导检查 | 全部代码单元格、嵌入的源码ZIP | 语法通过、SHA-256一致、恢复源码逐文件一致、再次运行不会覆盖已有修改。未在真实Colab网页/账号中执行 |

`part1_cpu/`可查看真实CPU图和耗时。其余名称含`smoke`或`synthetic`的目录仅用于验证程序能完整运行，不能当作学生正式实验结果。

## MRI数据审核

- Train：9,664张图像 + 9,664张标签，302位受试者。
- Validation：1,120张图像 + 1,120张标签，35位受试者。
- Test：544张图像 + 544张标签，17位受试者。
- 原图和标签均为256×256；训练/验证/测试受试者集合无交集。

## 尚未验证，必须实际运行

- CUDA显式DFT的设备执行与GPU时间。
- 真实LFW上的RF/CNN准确率及比较。
- 完整CIFAR-10训练达到>90%、>=94%和速度挑战。
- 完整MRI数据训练后的VAE表达质量、UNet每类DSC>0.9、GAN逼真度及无mode collapse。
- 真实Colab页面的账号连接、GPU配额和Google Drive授权。
- Rangpur登录、当前partition/module配置、GPU作业提交和现场演示。
- GitHub个人账号、真实提交历史及Git进阶短课。

完整包不包含声称达标的预训练权重。你运行正式训练后，真实模型、图片、指标存入`runs/`，可通过Notebook末尾下载。

本目录不会被readiness工具用来判定正式任务达标。

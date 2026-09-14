# COMP3710 Lab 2 — Pattern Recognition

Fourier analysis, face recognition, CIFAR-10 classification and MRI modelling using NumPy and PyTorch.

本项目包含傅里叶分析、人脸识别、图像分类，以及 MRI 重建、分割与生成。

## Notebook / 演示入口

[Open in Google Colab](https://colab.research.google.com/github/claire12-liao/COMP3710-Lab2/blob/main/COMP3710_Lab2_Colab.ipynb)

The notebook contains explanations and recorded outputs.
Live model demonstrations require the saved checkpoints and datasets, which are stored separately in Google Drive.

Notebook 包含说明和运行输出。现场运行模型需要先恢复 Drive 中保存的模型权重和数据；仅下载本仓库并不包含这些文件。

## Recorded results / 实验结果

Experiments were run in Google Colab using a Tesla T4 GPU.

| Task | Recorded result |
|---|---|
| Fourier DFT | Implementations agreed within the numerical tolerance used |
| PCA + Random Forest, LFW | Test accuracy: 64.91% |
| Two-convolution CNN, LFW | Test accuracy: 61.80% |
| ResNet18, CIFAR-10 | Test accuracy: 95.24%; selected epoch: 99 |
| MRI VAE | Test reconstruction MSE: 0.002089; selected epoch: 29 |
| MRI U-Net | Per-class test Dice: 0.9981, 0.9485, 0.9600, 0.9764 |
| MRI GAN | 100 training epochs; generated samples and interpolation included |

Results and figures are stored in `runs/`.

## Interpretation and limitations / 结果分析与局限

- The face CNN did not outperform the PCA–Random Forest baseline.
  人脸 CNN 本次表现低于随机森林，部分人物类别的召回率较低。
- CIFAR-10 training took approximately 1,325 seconds on the T4.
  Accuracy exceeded 94%, but the 360-second training target was not achieved.
  测试准确率达到要求，但不能将短时间推理当作训练速度达标。
- U-Net training was interrupted after five completed epochs; the saved
  checkpoint from epoch 3 was used for evaluation.
  U-Net 使用验证集选择的第 3 轮模型，并非训练了 80 轮。
- GAN losses and thumbnail distances do not establish image realism
  or rule out memorisation. Generated images require visual assessment.
  GAN 的真实性、多样性及是否记忆训练样本，需要结合图片进一步判断。
- Colab results are not evidence of execution on Rangpur.
  Colab 运行记录不能代替 Rangpur 集群运行记录。

## Repository structure / 文件结构

- `lab/`: model definitions and experiment implementations
- `runs/`: recorded metrics, training histories and figures
- `docs/`: demo guide, sources, score mapping and Rangpur notes
- `cluster/`: Slurm submission script
- `tools/`: environment and project checks
- `requirements.txt`: Python dependencies
- `AI_USAGE.md`: AI assistance disclosure
- `verification/`: implementation validation notes

Datasets, virtual environments and model checkpoints are not committed.
数据集、虚拟环境和模型权重不上传到本仓库。

## AI assistance / AI 使用说明

AI assisted with implementation, explanations and troubleshooting.
See `AI_USAGE.md` for the disclosure and `docs/SOURCES.md` for references.

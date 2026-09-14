# Rangpur：必须由你使用UQ账号完成的环节

实验单第10页要求：在Demo中，ResNet-18必须在Rangpur上运行inference和一个训练epoch。第10页也指出除最低难度外，后续MRI任务需要使用集群。Colab用于学习和调试，不能自动替代这些要求。

本包没有你的UQ登录信息，也没有当前课程的partition、module或account配置。以下占位符只用于你必须填的集群配置，模型代码本身是完整实现。

## 1. 取得本学期的登录/队列说明

以Blackboard/Eds和实验单附录A的链接为准：
https://student.eait.uq.edu.au/infrastructure/compute/

在VS Code Terminal可以使用ssh连接，但不要把密码、密钥写入脚本或GitHub。

```bash
ssh YOUR_UQ_USERNAME@HOST_FROM_CURRENT_COURSE_INSTRUCTIONS
```

## 2. 上传项目和数据

上传整个解压项目到你有权限的目录，或从你自己的GitHub clone源码并单独上传数据/模型。进入包含`lab/`和`README.md`的根目录。课程共享MRI数据在`/home/groups/comp3710/`，需要根据实际子目录填写`--data`，不能假设本包预处理文件夹就在该目录根部。

如果你使用的是本包数据，`python -m lab.data`会检查成对文件和受试者划分。不要把其他共享格式直接当本包PNG目录。

## 3. 使用课程指定的GPU Python环境

按本学期说明加载环境。然后记录这个Python的完整路径：

```bash
which python
python -c "import torch; print(torch.__version__, torch.version.cuda)"
export LAB_PYTHON="$(which python)"
```

登录节点没有GPU并不代表GPU节点不可用。真正的`torch.cuda.is_available()`应在获得GPU allocation以后检查。不要在登录节点开始长时间训练。

## 4. 提交训练作业

根据课程给的实际队列和账户要求替换`YOUR_GPU_PARTITION`，必要时增加课程指定的`--account`等参数。不能原样照抄占位符。

```bash
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00 cluster/run_job.sh cifar
```

默认从项目根目录提交，脚本会使用`SLURM_SUBMIT_DIR`定位项目、`LAB_PYTHON`定位GPU环境。其他任务将末尾`cifar`换为`part1`、`faces`、`vae`、`unet`、`gan`；UNet/GAN可能需要更多wall time，按课程配额调整。

```bash
squeue -u "$USER"
```

查看本次job生成的Slurm日志，等训练结束确认`runs/cifar/resnet18_best.pt`存在且`metrics.json`中真实准确率达标。

## 5. Demo：使用已经训练好的模型，现场推理+完整epoch

提前按课程说明请求一个可在practical时间使用的GPU交互allocation。登录GPU节点，在项目根目录运行：

```bash
"$LAB_PYTHON" -m lab.cifar --mode demo --device cuda --require-slurm
```

这会载入最佳模型，做完整测试集推理，再对本项目固定45,000张训练图执行一个完整epoch。保存`runs/cifar/live_demo.json`，记录GPU名、hostname、SLURM_JOB_ID、样本数和耗时。验证集来自训练来源的另外5,000张，未作为训练样本。

此命令的Slurm检测是防止误在本机运行；tutor仍需看到你是在真实Rangpur。Colab环境不应伪造Slurm环境变量。

UNet现场推理：

```bash
"$LAB_PYTHON" -m lab.mri unet --mode demo --device cuda
```

显示`segmentations.png`和四类Dice，同时解释one-hot、skip connections和数据划分。网络或排队可能影响现场操作，因此保留图片和日志备份，但备份不能替代实验单要求的现场运行。

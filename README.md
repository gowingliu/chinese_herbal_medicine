# 中医药识别挑战赛

## 1. 概述
- 本方案旨在通过人工智能技术实现对中医药图片的自动识别与分类。  
- 基于ConvNeXt网络模型，结合五折交叉验证策略进行训练，并在训练过程中引入指数移动平均（Exponential Moving Average, EMA）技术以优化模型收敛性与稳定性，最终生成五个独立模型。  
- 在测试阶段，应用测试时增强（Test-Time Augmentation, TTA）策略提升预测鲁棒性，并将五个模型的预测结果进行加权平均融合，从而输出最终的分类类别。

## 2. 算法与模型说明
- **整体框架:** 
  - ConvNeXt由ResNet50发展而来，通过学习Swin Transformer的架构以及优化策略，在纯卷积神经网络上比Swin Transformer拥有更快的推理速度以及更高的准确率。
- **训练步骤**
  - 使用预训练模型为在imagenet22k上预训练，imgenet1k上微调的模型。
  - 通过五折交叉验证策略训练ConvNeXt-L。
  - 采取学习率分层策略。每层学习率为基础学习率的固定倍率
    - stages.0: 0.1,
    - stages.1: 0.25,
    - stages.2: 0.5,
    - stages.3: 0.75,
    - head    : 1.0
  - 训练阶段固定超参数：IMG_SIZE = 384, batch_size=8, BASE_LR = 5e-5。
- **测试步骤**
  - 对图像采取TTA增强策略：
    - 对图片采取原图和翻转操作，得到两张图片。
    - 从中裁剪出五个视角（中心和四角），增加模型看到的图像多样性。
    - 模型从两张图片共计十个视角得到的结果取平均值，作为模型的输出结果。
  - 对五折交叉验证策略训练得到的5个模型分别进行预测，得到的结果平均后输出。
 
## 3. 数据处理流程
- **训练集**
  - 进行数据增强。
    - 以50%概率执行图像水平镜像翻转。
    - 以20%概率执行图像垂直翻转。
    - RandAugment自动增强
    - 随机区域擦除
- **验证集**
  - 不进行数据增强。

## 4. 训练与复现流程
- **Dependencies:** This codebase was tested with the following environment configurations. It may work with other versions.
  - Ubuntu 22.04
  - CUDA 12.0
  - Python 3.11
  - PyTorch 2.1.0 + cu121    

- **Evaluate Pretrained Models:**  
训练模型链接: https://pan.baidu.com/s/1e970J6Uwv1ooLqmXb7-8aA?pwd=qapf 提取码: qapf  
将模型权重保存至user_data当中。  
```
cd code
# Make sure the script has execute permission
chmod +x test.sh
# Execute the test script
./test.sh
```

- **Training**  
Load the pre-trained file by default.  
```
cd code
# Make sure the script has execute permission
chmod +x train.sh
# Execute the train script
./train.sh
```

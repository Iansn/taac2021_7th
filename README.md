# 方案介绍

2021腾讯广告算法大赛赛道一第七名解决方案，主要分为三步，第一步利用分幕的groundtruth预训练一个视频分类网络这里分类网络我们选择了slowonly网络，第二步基于预训练好的slowonly网络抽取视频特征，用这个视频特征训练场景分割模型，场景分割模型我们是基于bmn改造的，第三步再使用之前的视频特征和文本特征训练一个网络作为最后的分类网络

# 算法依赖

pytorch==1.7.1

torchvison==0.8.2

cuda10.1

mmcv==1.2.5（建议安装官方根据cuda版本编译好的版本）

decord==0.6.0

 transformers==4.7.0

# 目录结构说明

├─data
├─result
├─structuring
│  ├─model
│  ├─pre
│  ├─pretrain_models
│  └─src
│      ├─data
│      ├─loss
│      ├─models
│      └─utils
├─tagging
│  ├─model
│  ├─pre
│  ├─pretrain_models
│  └─src
│      ├─data
│      ├─models
│      └─utils
└─temp_data

代码分为五个部分，其中data是指数据集文件，请将algo-2021数据集拷贝到这个文件中来；result文件夹里会存放最终的结果文件;tagging里面包括了分类预训练和最后分类的代码；structuring里面包含的是场景分割的代码；temp_data文件存放训练测试过程中的特征和中间的临时结果文件

tagging和structuring文件夹里都包含有model，pre,pretrain_models，src四个部分，其中model存放训练生成的模型，pre存放提取特征和一些预处理的代码，pretrain_models存放预训练模型，src存放模型，loss，数据集等的源代码

# 训练步骤

训练步骤如下：

1、下载预训练模型，slowonly网络的初始化我们是利用的mmaction2里面Kinetics-400上预训练的模型，所以需要先下载预训练的slowonly模型（https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb_20210308-e8dd9e82.pth），并将下载好后的模型放入./tagging/pretrain_models中

另外我们还用到了文本特征，所以需要下载bert模型，我们选用的是chinese-roberta-wwm-ext模型，模型可以到huggingface官方下载（https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main），也可以到[网盘](链接：https://pan.baidu.com/s/1ZwOg86MwlWsixCkLIKJSLQ 
提取码：abcd)下载，如果是网盘下载，将下载好的模型放入./tagging/pretrain_models中，直接解压就好，如果官方下载，请在./tagging/pretrain_models中创建一个chinese-roberta-wwm-ext的文件，将下载内容放入这个文件中，且下载的文件的命名方式请保证和官方的一致

2、训练模型，运行下面脚本即可（从预训练到最后训练完成，大概需要20小时左右）

sh train.sh



#  测试步骤

直接运行下面脚本(包含特征提取和预测最后结果，提取特征大概需要4-5小时，预测1-2小时)

sh inference.sh

最终生成结果在result文件中# taac2021_7th

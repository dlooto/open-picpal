# OpenPicPal

<br>

[中文文档](README_CN.md) | [English README](README.md)

<br>

OpenPicPal是一个用于图片训练和自动分类的开源软件项目。

作为一个Python项目，OpenPicPal基于InceptionV3基础模型，可以在使用者自身准备的图片数据集上训练出需要的图片分类模型，然后基于结果模型进行图片分类。

本图片分类项目，包含以下关键步骤：
1. 选择一个基础模型：如本项目以InceptionV3为基础模型（考虑到其在准确率和运行性能上的均衡，参数不算多等）
2. 准备图片数据：包括训练、验证、测试等所需的图片集
3. 训练模型：生成需要的结果模型
4. 图片分类/预测：使用结果模型进行分类/预测

## 数据准备
图片数据来源，可根据自身业务需要，从互联网上搜索开源数据集或自行写爬虫获取。

将准备好的图片数据，存入以下目录结构中:
```
data/
    |__saved_model/     # 训练结束后的模型文件存放目录
    |__train/           # 训练图片集存入此目录下的各子目录中
        |__book             
        |__cat              
        |__digit            
        |__movie            
        ……
    |__validation/      # 验证图片集, 注意保持与train目录中同样的子目录结构
```    

#### 图片集配置
* `train`各子目录中存放对应类别的图片文件，如`book/`目录下存放所有关于图书的图片
* 有多少个分类就要准备多少个子目录的图片
* 从`train`各子目录里挑选约1/4 ~ 1/3的图片数量放入`validation`对应子目录里。如`train/book/`中有100张图片，则可以考虑从中取出25张图片放入`validation/book/`目录，形成验证图片集。
* 应确保`validation`子目录数量与`train`中子目录数量一致，且维持一定的比例关系（如保持验证集图片量为训练集图片量的1/4 ~ 1/3）
* 一般来说，图片量越大分类效果越好，但考虑到训练时间及性能，应根据业务情况选择图片量
* 如果有新的分类，则增加新的子目录放于`train`和`validation`里

## 开发环境
0. 必须的程序包：
```shell
    python==3.9.2
    keras==2.11.0
    tensorflow==2.11.0
```
1. Clone代码：`git clone git@github.com:dlooto/open-picpal.git`
2. 切换到项目根目录: `cd open-picpal`
3. 创建上述"数据准备"中的data目录结构，准备好图片数据集. 其中`data/saved_model/`为训练结束后的模型文件存放目录。
4. 创建并修改配置文件:

```shell
cp open-picpal/config.py.example  open-picpal/config.py     # 拷贝示例文件
vi open-picpal/config.py         # 修改相关参数
```

5. pip安装python相关的库
```
pip install -r requirements.txt 
```

## 训练与分类

1. 设置分类标签
```python
# picpal/config.py
CLASS_LABELS = {
    0: "book",
    1: "cat",
    2: "digit",
    3: "movie",
}
```
* 确保config.py中`CLASS_LABELS`配置与`train`和`validation`中子目录名一致，比如你可以修改标签"book"为"books", 但应确保train、validation里的子目录也一并修改。

2. 修改训练参数：
```python
MODEL_FILE_NAME = 'new_model.h5'    # 图片分类时所用的模型文件名    

IMG_WIDTH, IMG_HEIGHT = 256, 256     
BATCH_SIZE = 32                      
EPOCHS = 20                         
```
* 其中，参数设置`(IMG_WIDTH, IMG_HEIGHT)`是InceptionV3模型所要求的输入图片尺寸, 你需要根据自身业务需求调整该尺寸。比如，你的业务中若都是手机尺寸图片，即图片高度远大于宽度，则你需要将设置`IMG_HEIGHT`大于`IMG_WIDTH`（或为`IMG_WIDTH`的倍数）
* `EPOCHS` 参数表示训练过程中数据集的迭代次数。例如，如果将`EPOCHS` 设置为 10，那么训练将在整个数据集上迭代 10 次。
* 在每个epoch 中，数据集通常被分成多个批次进行处理。`BATCH_SIZE` 参数表示每个批次中包含的样本数量。
* `EPOCHS`和`BATCH_SIZE`的不同设置，会影响训练时长；
* 模型训练完成后，你需求修改`MODEL_FILE_NAME`为最新生成的模型文件名，然后进行图片分类

3. 训练模型
```shell
python -m picpal.train
```

4. 图片分类
```shell
python -m picpal.predict
```

## 作为程序库使用
你也可以将OpenPicPal作为一个python库在其他业务代码里使用.
1. 打包发布： 
```
python setup.py sdist bdist_wheel
``` 
2. 在dist目录中找到生成的程序包(如`open-picpal-0.1.0.tar.gz`)并安装：
```
pip install dist/open-picpal-0.1.0.tar.gz
```
3. 在业务代码中使用：
```python
from picpal.train import Trainer

# 训练模型
trainer = Trainer(
    epochs=20,
    batch_size=32, 
    img_width=256, 
    img_height=256
)
trainer.train()


# 图片分类
from picpal.config import *
from picpal.predict import Predictor

predictor = Predictor(
    model_path=get_model_path(),
    class_labels=CLASS_LABELS,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT
)

img_path_list = [
    "cat/14694fdd.png",
    "movie/e08b23f.png",
    "book/33h07bu31.jpg",
]

for i in img_path_list:
    result = predictor.predict_with_image_path(
        get_validation_img_path(i)
    )
    print(result, i)
```

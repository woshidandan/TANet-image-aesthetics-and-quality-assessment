[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

这是我们组在IJCAI 2022关于图像美学评估最新的一篇工作，因我个人热衷于开源，希望更多的人能够关注到这篇工作，所以另外写了一篇中文的介绍给国内的小伙伴: 

<div align="center">
<h1>
<b>
Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks
</b>
</h1>
<h4>
<b>
Shuai He, Yongchang Zhang, Dongxiang Jiang, Rui Xie, Anlong Ming
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

<!-- ![TANet and TAD66K dataset](https://user-images.githubusercontent.com/15050507/164587655-4af0b519-7213-4f29-b378-5dfc51dfab83.png)
![Performance](https://user-images.githubusercontent.com/15050507/164587663-043a76d8-5d1b-417e-856d-2320fbe26836.png) -->
------------------------------------------------------------------------------------------------------------

# TAD66K数据集 &nbsp;<a href=""><img width="48" src="docs/release_icon.png"></a>

## 介绍
* 简要版：一个新的美学数据集，6万6千张左右图像，按主题分类标注。
* 太长不看版：以主题为核心，以开源为理念，我们建立了一个包含6万6千张左右图像的数据集，可用于图像美学评估。建立这样一个数据集的初衷，源自于组内同学在标注图像美感时的困惑，我们如何去评价一朵花和一个人之间美感的区别呢？显然，不同主题的图片，通常包含了不同的评分规则，标注人员在标注图像的过程中，隐性的会考虑到当前图像的主题，但现有的数据集，通常将所有类别的图像混合在一起进行标注，事实上，这可能会引入大量的噪声。因此，我们通过半年多的时间，收集，整理和标注了一批图片，包含了47种常见的主题，每个主题包含1千张以上的图像，各个主题的图像分开标注，每张图像至少被1200以上的人浏览和评价过，计算出平均分作为分数。

![TAD66K](https://user-images.githubusercontent.com/15050507/164620789-2958fbd6-5e3b-4eba-9697-bcd28d5257f6.png)

<div align="center">
    
![example3](https://user-images.githubusercontent.com/15050507/164624400-acb365e0-05d9-4de9-bc16-f894904c6d33.png)
    
</div>

## Download
* 你可以从这里下载到数据集和标注分数 [here](https://drive.google.com/drive/folders/1lpSqNXtm-ianfI6TIvrJZGp96iCXsBR-)，如果被墙了，记得cue我，每张图像我们都将最大边按等比放缩至800。

------------------------------------------------------------------------------------------------------------

# TANet网络 &nbsp;<a href=""><img width="48" src="docs/release_icon.png"></a>

## 介绍
* 简要版：在通用美学数据集AVA，个性化美学数据集FLICKR-AES, 以及自建的数据集TAD66K，全SOTA。
* 太长不看版：我们提出了一个以主题为核心的网络架构TANet，在搭建这个网络的过程中，我们希望其能够切实的提取出当前图像的主题，因此将一个百万级别的数据集Place用来预训练我们其中的一个分支。Place数据集包含了大部分现实场景，虽然场景无法直接等效于主题，但据我们所知，这是目前最好的能进行主题感知的方法。值得注意的是，我们发现经过预训练的分支会出现注意力弥散现象，这会导致费尽力气预训练获得的主题感知能力丧失，这一点在此前用ImageNet进行预训练的工作中也有体现，因此我们会将该分支直接冻结。为了让网络能够自适应的利用主题信息，融合的权重是其学习得到的；为了能够让其获得图像中不同区域色彩的分布及关系信息，我们专门加了一个类似自注意力机制的分支。

![TANet](https://user-images.githubusercontent.com/15050507/164627140-fed5f9b9-43fa-4cb3-a23f-b60935d3aa71.png)
![Performance](https://user-images.githubusercontent.com/15050507/164587663-043a76d8-5d1b-417e-856d-2320fbe26836.png)


## 代码环境
* pandas==0.22.0
* nni==1.8
* requests==2.18.4
* torchvision==0.8.2+cu101
* numpy==1.13.3
* scipy==0.19.1
* tqdm==4.43.0
* torch==1.7.1+cu101
* scikit_learn==1.0.2
* tensorboardX==2.5

## 如何训练和测试
* 炼丹是一个痛苦的过程，特别是像TANet这种的多分支网络，每个分支若设置相同的学习率，训练起来无法达到最优的性能，若单独手工设置各分支学习率，耗时又耗力，所以这里面我们用了微软的自动调参工具[nni](https://github.com/microsoft/nni)，网上有很多nni相关的[使用教程](https://blog.csdn.net/weixin_43653494/article/details/101039198)，强烈推荐同学们使用这个工具，不仅能自动调参，还能替代TensorBoard对训练过程的各项指标可视化。
* 如果你安装好了nni之后，训练时请配置好config.yml和超参数文件search_space.json，然后运行nnictl create --config config.yml -p 8999，训练的可视化后台可以在本地的http://127.0.0.1:8999 或 http://172.17.0.3:8999 看到。
* 如果你不想用这个工具训练或测试，只需要将代码中类似于param_group['lr']这样的超参数的中括号都改为param_group.lr就可以了。
* PS：FLICKR-AES这个数据集上train的权重可能不会公开，因为目前和一个公司合作，有一些保密的需求。

## 快来尝试一下吧!
https://user-images.githubusercontent.com/15050507/164580816-f98d1dd9-50a0-47b7-b992-2f0374e8a418.mp4

https://user-images.githubusercontent.com/15050507/164580823-4ea8ff91-825b-43dc-a421-f75455e549ae.mp4

https://user-images.githubusercontent.com/15050507/164580840-b7f5624f-486d-46e6-9dd4-efaa92dde09c.mp4

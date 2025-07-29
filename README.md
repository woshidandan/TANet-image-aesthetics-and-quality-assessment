[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

[[国内的小伙伴请看更详细的中文说明]](https://github.com/woshidandan/TANet/blob/main/README_CN.md)This repo contains the official implementation and the new IAA dataset TAD66K of the **IJCAI 2022** paper. Our new work on ICCV2023：[Link](https://github.com/woshidandan/Image-Color-Aesthetics-Assessment)

<div align="center">
<h1>
<b>
Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks 
</b>
</h1>
<h4>
<b>
Shuai He, Yongchang Zhang, Rui Xie, Dongxiang Jiang, Anlong Ming
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

<!-- ![TANet and TAD66K dataset](https://user-images.githubusercontent.com/15050507/164587655-4af0b519-7213-4f29-b378-5dfc51dfab83.png)
![Performance](https://user-images.githubusercontent.com/15050507/164587663-043a76d8-5d1b-417e-856d-2320fbe26836.png) -->
------------------------------------------------------------------------------------------------------------

# TAD66K &nbsp;<a href=""><img width="48" src="docs/release_icon.png"></a>

## Introduction
* We build a large-scale dataset called the Theme and Aesthetics Dataset with 66K images (TAD66K), which is specifically designed for IAA. Specifically, (1) it is a theme-oriented dataset containing 66K images covering 47 popular themes. All images were carefully selected by hand based on the theme. (2) In addition to common aesthetic criteria, we provide 47 criteria for the 47 themes. Images of each theme are annotated independently, and each image contains at least 1200 effective annotations (so far the richest annotations). These high-quality annotations could help to provide deeper insight into the performance of models. 

![TAD66K](https://user-images.githubusercontent.com/15050507/164620789-2958fbd6-5e3b-4eba-9697-bcd28d5257f6.png)

<div align="center">
    
![example3](https://user-images.githubusercontent.com/15050507/164624400-acb365e0-05d9-4de9-bc16-f894904c6d33.png)
    
</div>

## Download Dataset
* Download from [here google](https://drive.google.com/drive/folders/1b2D9LeeG5XZzhEa8ldnIZjGh0IHadHhU?usp=sharing), it contains images with the largest side scaled to 800, and labels categorized by different themes.
* or [here baidu](https://pan.baidu.com/s/1bAiDMwKLF_vLZKelz5ZfRg), code: 8888 

------------------------------------------------------------------------------------------------------------

# TANet &nbsp;<a href=""><img width="48" src="docs/release_icon.png"></a>

## Introduction
We propose a baseline model, called the Theme and Aesthetics Network (TANet), which can maintain a constant perception of aesthetics to effectively deal with the problem of attention dispersion. Moreover, TANet can adaptively learn the rules for predicting aesthetics according to a recognized theme. By comparing 17 methods with TANet on three representative datasets: AVA, FLICKR-AES and the proposed TAD66K, TANet achieves state-of-the-art performance on all three datasets.
![TANet](https://user-images.githubusercontent.com/15050507/164627140-fed5f9b9-43fa-4cb3-a23f-b60935d3aa71.png)
![Performance](https://user-images.githubusercontent.com/15050507/164587663-043a76d8-5d1b-417e-856d-2320fbe26836.png)


## Environment Installation
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

## How to Run the Code
* We used the hyperparameter tuning tool [nni](https://github.com/microsoft/nni), maybe you should know how to use this tool first (it will only take a few minutes of your time), because our training and testing will be in this tool.
* Train or test, please run: nnictl create --config config.yml -p 8999
* The Web UI urls are: http://127.0.0.1:8999 or http://172.17.0.3:8999
* Note: nni is not necessary, if you don't want to use this tool, just make simple modifications to our code, such as changing param_group['lr'] to param_group.lr, etc.
* PS: The work of train on the FLICKR-AES dataset may not be made public, because we are currently cooperating with a company, and the relevant model has been embedded into the system, and there are some confidentiality requirements.

## If you find our work is useful, pleaes cite our paper:
```
@article{herethinking,
  title={Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks},
  author={He, Shuai and Zhang, Yongchang and Xie, Rui and Jiang, Dongxiang and Ming, Anlong},
  journal={IJCAI},
  year={2022},
}
```

## Try!
https://user-images.githubusercontent.com/15050507/164580816-f98d1dd9-50a0-47b7-b992-2f0374e8a418.mp4

https://user-images.githubusercontent.com/15050507/164580823-4ea8ff91-825b-43dc-a421-f75455e549ae.mp4

https://user-images.githubusercontent.com/15050507/164580840-b7f5624f-486d-46e6-9dd4-efaa92dde09c.mp4

## Related Work from Our Group
<table>
  <thead align="center">
    <tr>
      <td><b>🎁 Projects</b></td>
      <td><b>📚 Publication</b></td>
      <td><b>🌈 Content</b></td>
      <td><b>⭐ Stars</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/woshidandan/Attacker-against-image-aesthetics-assessment-model"><b>Attacker Against IAA Model【美学模型的攻击和安全评估框架】</b></a></td>
      <td><b>TIP 2025</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Attacker-against-image-aesthetics-assessment-model?style=flat-square&labelColor=343b41"/></td>
    </tr
    <tr>
      <td><a href="https://github.com/woshidandan/Rethinking-Personalized-Aesthetics-Assessment"><b>Personalized Aesthetics Assessment【个性化美学评估新范式】</b></a></td>
      <td><b>CVPR 2025</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Rethinking-Personalized-Aesthetics-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment"><b>Pixel-level image exposure assessment【首个像素级曝光评估】</b></a></td>
      <td><b>NIPS 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment"><b>Long-tail solution for image aesthetics assessment【美学评估数据不平衡解决方案】</b></a></td>
      <td><b>ICML 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Prompt-DeT"><b>CLIP-based image aesthetics assessment【基于CLIP多因素色彩美学评估】</b></a></td>
      <td><b>Information Fusion 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Prompt-DeT?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment"><b>Compare-based image aesthetics assessment【基于对比学习的多因素美学评估】</b></a></td>
      <td><b>ACMMM 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment"><b>Image color aesthetics assessment【首个色彩美学评估】</b></a></td>
      <td><b>ICCV 2023</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Aesthetics-and-Quality-Assessment"><b>Image aesthetics assessment【通用美学评估】</b></a></td>
      <td><b>ACMMM 2023</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment"><b>Theme-oriented image aesthetics assessment【首个多主题美学评估】</b></a></td>
      <td><b>IJCAI 2022</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/TANet-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/AK4Prompts"><b>Select prompt based on image aesthetics assessment【基于美学评估的提示词筛选】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/AK4Prompts?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/mRobotit/M2Beats"><b>Motion rhythm synchronization with beats【动作与韵律对齐】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/mRobotit/M2Beats?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC"><b>Champion Solution for AIGC Image Quality Assessment【NTIRE AIGC图像质量评估赛道冠军】</b></a></td>
      <td><b>CVPRW NTIRE 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC?style=flat-square&labelColor=343b41"/></td>
    </tr>
  </tbody>
</table>

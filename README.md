[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repo contains the official implementation and the new IAA dataset TAD66K of the **IJCAI 2022** paper: 

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

![TANet and TAD66K dataset](https://user-images.githubusercontent.com/15050507/164587655-4af0b519-7213-4f29-b378-5dfc51dfab83.png)
![Performance](https://user-images.githubusercontent.com/15050507/164587663-043a76d8-5d1b-417e-856d-2320fbe26836.png)

# TAD66K
* We build a large-scale dataset called the Theme and Aesthetics Dataset with 66K images (TAD66K), which is specifically designed for IAA. Specifically, (1) it is a theme-oriented dataset containing 66K images covering 47 popular themes. All images were carefully selected by hand based on the theme. (2) In addition to common aesthetic criteria, we provide 47 criteria for the 47 themes. Images of each theme are annotated independently, and each image contains at least 1200 effective annotations (so far the richest annotations). These high-quality annotations could help to provide deeper insight into the performance of models. 

![TAD66K](https://user-images.githubusercontent.com/15050507/164620789-2958fbd6-5e3b-4eba-9697-bcd28d5257f6.png)



We propose a baseline model, called the Theme and Aesthetics Network (TANet), which can maintain a constant perception of aesthetics to effectively deal with the problem of attention dispersion. Moreover, TANet can adaptively learn the rules for predicting aesthetics according to a recognized theme. TAD

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

## TAD66K
* Download from [here](https://drive.google.com/drive/folders/1lpSqNXtm-ianfI6TIvrJZGp96iCXsBR-).




https://user-images.githubusercontent.com/15050507/164580816-f98d1dd9-50a0-47b7-b992-2f0374e8a418.mp4

https://user-images.githubusercontent.com/15050507/164580823-4ea8ff91-825b-43dc-a421-f75455e549ae.mp4

https://user-images.githubusercontent.com/15050507/164580840-b7f5624f-486d-46e6-9dd4-efaa92dde09c.mp4

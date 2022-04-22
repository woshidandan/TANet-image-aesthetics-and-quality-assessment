[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repo contains the official implementation of the **IJCAI 2022** paper: 

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

https://user-images.githubusercontent.com/15050507/164580816-f98d1dd9-50a0-47b7-b992-2f0374e8a418.mp4

https://user-images.githubusercontent.com/15050507/164580823-4ea8ff91-825b-43dc-a421-f75455e549ae.mp4

https://user-images.githubusercontent.com/15050507/164580840-b7f5624f-486d-46e6-9dd4-efaa92dde09c.mp4

## Environment Installation
*pandas==0.22.0
*nni==1.8
requests==2.18.4
torchvision==0.8.2+cu101
numpy==1.13.3
scipy==0.19.1
tqdm==4.43.0
torch==1.7.1+cu101
scikit_learn==1.0.2
tensorboardX==2.5

# How to Run the Code
1. We used the hyperparameter tuning tool [nni](https://github.com/microsoft/nni), maybe you should know how to use this tool first (it will only take a few minutes of your time), because our training and testing will be in this tool. 我们使用了 hyperparameter tuning工具，或许你应该先了解如何使用这个工具（仅会花费你几分钟的时间），因为我们的训练和测试会在这个工具中。
2. Train or test, please run: nnictl create --config config.yml -p 8999
3. The Web UI urls are: http://127.0.0.1:8999 or http://172.17.0.3:8999
Note: nni is not necessary, if you don't want to use this tool, just make simple modifications to our code, such as changing param_group['lr'] to param_group.lr, etc.nni并不是必要的，如果你不想使用这个工具，只需要对我们的代码进行简单的修改，例如将param_group['lr']改为param_group.lr等等

# TAD66K
Download from [here](https://drive.google.com/drive/folders/1lpSqNXtm-ianfI6TIvrJZGp96iCXsBR-).

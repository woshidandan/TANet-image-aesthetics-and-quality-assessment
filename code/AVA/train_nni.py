from __future__ import print_function, division
import os
import torch
import numpy as np
import math
import torch.optim as optim
import option
import nni
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import models
from dataset import AVADataset
from util import EDMLoss, AverageMeter
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from nni.utils import merge_parameter
opt = option.init()
device = torch.device("cuda:0")
MOBILE_NET_V2_UTR = 'https://s3-us-west-1.amazonaws.com/models-nima/mobilenetv2.pth.tar'

def adjust_learning_rate(params, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = params.init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # avgpool
        self.avgpool = nn.AvgPool2d(input_size // 32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def resnet365_backbone():
    arch = 'resnet18'
    model_file = './resnet18_places365.pth.tar'
    last_model = models.__dict__[arch](num_classes=365)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    last_model.load_state_dict(state_dict)

    return last_model

def mobile_net_v2(pretrained=False):
    model = MobileNetV2()
    if pretrained:
        print("read mobilenet weights")
        path_to_model = '/root/tmp/pycharm_project_815/M_M_Semi-Supervised/code/AVA/pretrain_model/mobilenetv2.pth.tar'
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    return model

def Attention(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))
    return sim_map

def MV2():
    model = mobile_net_v2()
    model = nn.Sequential(*list(model.children())[:-1])
    # model_dict = model.state_dict()
    return model

class L5(nn.Module):
    def __init__(self):
        super(L5, self).__init__()
        back_model = MV2()
        self.base_model = back_model
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

        self.last_out_w = nn.Linear(365, 100)
        self.last_out_b = nn.Linear(365, 1)
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, x):
        res_last_out_w = self.last_out_w(x)
        res_last_out_b = self.last_out_b(x)
        param_out = {}
        param_out['res_last_out_w'] = res_last_out_w
        param_out['res_last_out_b'] = res_last_out_b
        return param_out

# L3
class TargetNet(nn.Module):
    def __init__(self):
        super(TargetNet, self).__init__()

        # L2
        self.fc1 = nn.Linear(365, 100)
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)
        self.bn1 = nn.BatchNorm1d(100).cuda()
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(1 - 0.5)

        self.relu7 = nn.PReLU()
        self.relu7.cuda()
        self.sig = nn.Sigmoid()

    def forward(self, x, paras):
        q = self.fc1(x)
        q = self.bn1(q)
        q = self.relu1(q)
        q = self.drop1(q)

        self.lin = nn.Sequential(TargetFC(paras['res_last_out_w'], paras['res_last_out_b']))
        q = self.lin(q)
        bn7 = nn.BatchNorm1d(q.shape[0])
        bn7.cuda()
        q = bn7(q)
        q = self.relu7(q)

        return q

class TargetFC(nn.Module):
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        out = F.linear(input_, self.weight, self.bias)
        return out

class TANet(nn.Module):
    def __init__(self):
        super(TANet, self).__init__()
        self.res365_last = resnet365_backbone()
        self.hypernet = L1()

        # L3
        self.tygertnet = TargetNet()

        self.avg = nn.AdaptiveAvgPool2d((10, 1))
        self.avg_RGB = nn.AdaptiveAvgPool2d((12, 12))

        self.mobileNet = L5()
        self.softmax = nn.Softmax(dim=1)

        # L4
        self.head_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(20736, 10),
            nn.Softmax(dim=1)
        )

        # L6
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(30, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x_temp = self.avg_RGB(x)
        x_temp = Attention(x_temp)
        x_temp = x_temp.view(x_temp.size(0), -1)
        x_temp = self.head_rgb(x_temp)

        res365_last_out = self.res365_last(x)
        res365_last_out_weights = self.hypernet(res365_last_out)
        res365_last_out_weights_mul_out = self.tygertnet(res365_last_out, res365_last_out_weights)

        x2 = res365_last_out_weights_mul_out.unsqueeze(dim=2)
        x2 = self.avg(x2)
        x2 = x2.squeeze(dim=2)

        x1 = self.mobileNet(x)
        x = torch.cat([x1, x2, x_temp], 1)
        x = self.head(x)
        return x

def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

def create_data_part(opt):
    train_csv_path = os.path.join(opt['path_to_save_csv'], 'train.csv')
    val_csv_path = os.path.join(opt['path_to_save_csv'], 'val.csv')
    test_csv_path = os.path.join(opt['path_to_save_csv'], 'test.csv')

    train_ds = AVADataset(train_csv_path, opt['path_to_images'], if_train=True)
    val_ds = AVADataset(val_csv_path, opt['path_to_images'], if_train=False)
    test_ds = AVADataset(test_csv_path, opt['path_to_images'], if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def train(opt, model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
    model.train()

    # Freeze
    for name, param in model.named_parameters():
        if name[:11] == "res365_last":
            param.requires_grad = False
        else:
            param.requires_grad = True

    train_losses = AverageMeter()
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.type(torch.FloatTensor).to(device)
        y = y.to(device).view(y.size(0), -1).float()
        y_pred = model(x).float()
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))
    return train_losses.avg

def validate(opt,model, loader, criterion, writer=None, global_step=None, name=None, test_or_valid_flag = 'test'):
    model.eval()
    validate_losses = AverageMeter()
    torch.set_printoptions(precision=3)
    true_score = []
    pred_score = []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.type(torch.FloatTensor).to(device)
        y = y.to(device).view(y.size(0), -1)
        y_pred = model(x)
        pscore, pscore_np = get_score(opt, y_pred)
        tscore, tscore_np = get_score(opt, y)
        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()
        loss = criterion(y_pred, y).float()
        validate_losses.update(loss.item(), x.size(0))

    lcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)
    print('{}, accuracy: {}, lcc_mean: {}, srcc_mean: {}, validate_losses: {}'.format(test_or_valid_flag, acc,
                                                                                      lcc_mean[0], srcc_mean[0],
                                                                                      validate_losses.avg))
    return validate_losses.avg, acc, lcc_mean, srcc_mean

def start_train(opt):
    dataloader_train, dataloader_valid, dataloader_test = create_data_part(opt)
    criterion = EDMLoss()
    criterion.to(device)
    model = TANet()

    model.load_state_dict(torch.load(opt['path_to_model_weight'], map_location='cuda:0'))
    model = model.to(device)

    optimizer = optim.Adam([
        # {'params': other_params},
        {'params': model.res365_last.parameters(), 'lr': opt['init_lr_res365_last']},
        {'params': model.mobileNet.parameters(), 'lr': opt['init_lr_mobileNet']},
        {'params': model.head.parameters(), 'lr': opt['init_lr_head']},
        {'params': model.head_rgb.parameters(), 'lr': opt['init_lr_head_rgb']},
        {'params': model.hypernet.parameters(), 'lr': opt['init_lr_hypernet']},
        {'params': model.tygertnet.parameters(), 'lr': opt['init_lr_tygertnet']},
    ], lr=opt['init_lr'])

    writer = SummaryWriter(log_dir=os.path.join(opt['experiment_dir_name'], 'logs'))
    srcc_best = 0
    vacc_best = 0

    for e in range(opt['num_epoch']):
        # please set util.py r = 2 of EMD
        # train_loss = train(opt,model=model, loader=dataloader_train, optimizer=optimizer, criterion=criterion,
        #                    writer=writer, global_step=len(dataloader_train) * e,
        #                    name=f"{opt['experiment_dir_name']}_by_batch")
        # val_loss,vacc,vlcc,vsrcc = validate(opt,model=model, loader=dataloader_valid, criterion=criterion,
        #                     writer=writer, global_step=len(dataloader_valid) * e,
        #                     name=f"{opt['experiment_dir_name']}_by_batch", test_or_valid_flag='valid')

        # please set util.py r = 1 of EMD
        test_loss, tacc, tlcc, tsrcc = validate(opt, model=model, loader=dataloader_test, criterion=criterion,
                                                writer=writer, global_step=len(dataloader_test) * e,
                                                name=f"{opt['experiment_dir_name']}_by_batch",
                                                test_or_valid_flag='test')
        nni.report_intermediate_result(
            {'default': tacc, "vsrcc": tsrcc[0], "val_loss": test_loss})
    nni.report_final_result({'default': tacc, "vsrcc": tsrcc[0]})
    writer.close()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    print(os.getcwd())
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(opt, tuner_params))
    print(params)
    start_train(params)

import os
import requests
import torch
import torch.nn as nn
Gl_z = torch.ones(64,10)

def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)

        cdf_diff = cdf_estimate - cdf_target
        # samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))  # train
        samplewise_emd = torch.mean(torch.pow(torch.abs(cdf_diff), 1)) # test

        return samplewise_emd.mean()
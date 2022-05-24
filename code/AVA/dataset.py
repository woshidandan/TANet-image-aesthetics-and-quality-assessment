import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

class AVADataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        scores_names = [f'score{i}' for i in range(2, 12)]
        y = np.array([row[k] for k in scores_names])

        p = y / y.sum()
        image_id = row['image_id']

        image_id = int(image_id)

        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)

        image = image.resize((224, 224))
        x = self.transform(image)

        return x, p.astype('float32')

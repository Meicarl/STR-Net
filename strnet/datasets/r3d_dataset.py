import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F


def normalize(intermediate_reconst, normalize_on_batch=False):
    mag = np.abs(intermediate_reconst)

    if normalize_on_batch:
        intermediate_reconst /= mag.max()
    else:
        intermediate_reconst /= np.max(mag, axis=(-1, -2, -3), keepdims=True)
    return intermediate_reconst

def addNoise(SNRdB,signal):
    def _addNoise(SNRdB,signal):
    
        signal_power = np.mean(np.abs(signal)**2)
        snr_lin = 10**(SNRdB/10)
        variance = signal_power/snr_lin
        std=np.sqrt(variance)

        nim = np.random.randn(*signal.shape)*1j*np.sqrt(1/2)
        nre = np.random.randn(*signal.shape)*np.sqrt(1/2)
        
        n=std*(nim+nre)

        return n+signal

    if len(signal.shape)>=4:
        noisy_signal=[]
        for i in tqdm(range(signal.shape[0])):
            noisy_signal.append(_addNoise(SNRdB,signal[i,...])[None,...])
        
        return np.concatenate(noisy_signal,0)
    else:
        return _addNoise(SNRdB,signal)

class NumpyDataset(Dataset):
    def __init__(self, data_file, label_file, data_transforms=None, label_transforms=None, normalize_on_batch=False):
        self.data = np.load(data_file)
        self.labels = np.load(label_file)
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.normalize_on_batch = normalize_on_batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.abs(self.data[idx])  # shape: (25, 25, 49)，取复数的模
        label = np.abs(self.labels[idx])  # shape: (25, 25, 49)，取复数的模

        # 为每条数据增加一个通道维度
        data = np.expand_dims(data, axis=0)  # 从 (25, 25, 49) 变为 (1, 25, 25, 49)
        label = np.expand_dims(label, axis=0)  # 从 (25, 25, 49) 变为 (1, 25, 25, 49)

        # 归一化
        data = normalize(data, normalize_on_batch=self.normalize_on_batch)
        label = normalize(label, normalize_on_batch=self.normalize_on_batch)

        if self.data_transforms is not None:
            data = self.data_transforms(data)
        if self.label_transforms is not None:
            label = self.label_transforms(label)

        return data, label


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float)

# PSNR 计算函数
def psnr(img1, img2):
    mse = torch.nn.functional.mse_loss(img1, img2)
    max_pixel_value = 1.0  # 假设图像像素值在 [0, 1] 范围内
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg  

if __name__ == '__main__':
    data_file = '/root/lanyun-tmp/data/x_train.npy'
    label_file = '/root/lanyun-tmp/data/y_train.npy'
    dataset = NumpyDataset(data_file, label_file, data_transforms=to_tensor, label_transforms=to_tensor, normalize_on_batch=False, cut=False, augment=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for data, label in dataloader:
        print(data.shape)
        print(label.shape)
        break

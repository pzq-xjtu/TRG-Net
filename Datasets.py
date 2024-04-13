import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from PIL import Image
from numpy.random import RandomState
import torchvision.transforms as transforms

from options import copy_opt_file

class Dataset(udata.Dataset):
    def __init__(self, name, gtname, patchsize=None, length=None, path=None):
        super().__init__()
        if path is not None:
            copy_opt_file(__file__, path)
        self.dataset = name
        self.gtdata=gtname
        self.patch_size=patchsize
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(self.dataset)
        self.gt_dir = os.path.join(self.gtdata)
        self.mat_files = os.listdir(self.root_dir)
        self.file_num = len(self.mat_files)
        if length is None:
            self.sample_num = self.file_num
        else:
            self.sample_num = length

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        O1 = cv2.imread(img_file)
        b, g, r = cv2.split(O1)
        O = cv2.merge([r, g, b])

        if self.patch_size is not None:
            O,row,col= self.crop(O)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.gt_dir, file_name)
        B = cv2.imread(gt_file)
        assert B.shape == O1.shape
        b, g, r = cv2.split(B)
        B = cv2.merge([r, g, b])
        if self.patch_size is not None:
            B = B[row: row + self.patch_size, col : col + self.patch_size]
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))

        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c


class DatasetReal(udata.Dataset):
    def __init__(self, rain_dir, norain_dir, patchsize=0, length=None, path=None):
        super().__init__()
        if path is not None:
            copy_opt_file(__file__, path)
        self.rain_dir = rain_dir
        self.norain_dir = norain_dir
        self.patch_size=patchsize
        self.rain_dir = os.path.join(self.rain_dir)
        self.norain_dir = os.path.join(self.norain_dir)
        self.rain_files = os.listdir(self.rain_dir)
        self.norain_files = os.listdir(self.norain_dir)
        self.rain_file_num = len(self.rain_files)
        self.norain_file_num = len(self.norain_files)
        if length is None:
            self.sample_num = self.norain_file_num
        else:
            self.sample_num = length

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        idx1 = random.randint(0, self.rain_file_num)
        rain_file_name = self.rain_files[idx1 % self.rain_file_num]
        rain_img_file = os.path.join(self.rain_dir, rain_file_name)
        O = cv2.imread(rain_img_file)
        O1 = O
        b, g, r = cv2.split(O)
        O = cv2.merge([r, g, b])
        if self.patch_size:
            O,row,col= self.crop(O)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))


        norain_file_name = self.norain_files[idx % self.norain_file_num]
        gt_file = os.path.join(self.norain_dir, norain_file_name)
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        B = cv2.merge([r, g, b])
        if self.patch_size:
            B,row,col= self.crop(B)
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))

        return torch.Tensor(O), torch.Tensor(B), norain_file_name

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if min(h, w) == p_h:
            r = c = 0
        else:
            r = random.randint(0, h - p_h)
            c = random.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c
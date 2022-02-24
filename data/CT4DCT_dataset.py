import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
from .transform import MultiTransform
from .gray_augmentaion import grayaug
import SimpleITK as sitk


class CT4DCTdataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "iDose")  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "IMR")

        filenames_A = set(os.listdir(self.dir_A))
        filenames_B = set(os.listdir(self.dir_B))

        self.filenames = list(filenames_A & filenames_B)

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        filename = self.filenames[index]
        A_path = os.path.join(self.dir_A, filename)
        B_path = os.path.join(self.dir_B, filename)

        A = Image.open(A_path).convert('I')
        B = Image.open(B_path).convert('I')

        A = np.array(A)
        B = np.array(B)

        while A.max() == 0 or B.max()==0:
            filename = self.filenames[(index+random.randint(1, len(self.filenames)-1))%(len(self.filenames))]
            A_path = os.path.join(self.dir_A, filename)
            B_path = os.path.join(self.dir_B, filename)
            A = Image.open(A_path).convert('I')
            B = Image.open(B_path).convert('I')
            A = np.array(A)
            B = np.array(B)

        # split AB image into A and B
        # apply the same transform to both A and B


        CT_min = -1024
        CT_max = 3072
        
        if self.opt.input_nbits == "uint16":
            A =  A.astype(np.float32)
            A = A - 1024                        # 还原成CT值
            A = A - (CT_max + CT_min)*0.5        # B  假设需要关注和保留的CT值范围为: [-1024,  3072], CT值精度为1
            A =  2.0 * A / (CT_max - CT_min)
            A = np.clip(A, -1, 1)               # 截断 非关注区域
        elif self.opt.input_nbits == "uint8":
            A = A.astype(np.float32)
            A = 255 * (A - CT_min) / (CT_max - CT_min)
            A = np.clip(A, 0, 255)
            A = A.astype(np.uint8)   # 截断成 8位图   , 有效CT范围为： CT： [-1024,  3072], CT值精度为4096/255 = 16
            A = A.astype(np.float32)
            A = A - 127.0       #
            A = A / 127.0      # 截断 非关注区域

        if self.opt.output_nbits == "uint16":
            B =  B.astype(np.float32)
            B = B - 1024                        # 还原成CT值
            B = B - (CT_max + CT_min)*0.5        # B  假设需要关注和保留的CT值范围为: [-1024,  3072], CT值精度为1
            B =  2.0 * B / (CT_max - CT_min)
            B = np.clip(B, -1, 1)               # 截断 非关注区域
        elif self.opt.output_nbits == "uint8":
            B = B.astype(np.float32)
            B = 255 * (B - CT_min) / (CT_max - CT_min)
            B = np.clip(B, 0, 255)
            B = B.astype(np.uint8)   # 截断成 8位图   , 有效CT范围为： CT： [-1024,  3072], CT值精度为4096/255 = 16
            B = B.astype(np.float32)
            B = B - 127.0       #
            B = B / 127.0      # 截断 非关注区域

       #  if self.opt.phase == "train":
            # [A, B] = grayaug(A, B , [-1, 1])
       #      [A, B] = MultiTransform(A, B)

       #  A = A.astype(np.float32)
      #   B = B.astype(np.float32)

        # 将矩阵转成tensor
        A = torch.from_numpy(A)
        B = torch.from_numpy(B)
        A = A.unsqueeze(0)   # 补充一位 channel维度
        B = B.unsqueeze(0)


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.filenames)



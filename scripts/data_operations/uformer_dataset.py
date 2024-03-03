import os
from turtle import color
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class Image_Dataset(Dataset):
    def __init__(self, root_dir, color_format = 'rgb', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.input_rgb_dir = os.path.join(root_dir, 'input')
        self.target_dir = os.path.join(root_dir, 'gt')

        print(self.input_rgb_dir, self.target_dir)

        self.color_format = color_format

        self.input_rgb_images = sorted(os.listdir(self.input_rgb_dir))
        self.target_images = sorted(os.listdir(self.target_dir))

    def __len__(self):
        return len(self.input_rgb_images)

    def __getitem__(self, idx):

        input_image_rgb_name = os.path.join(self.input_rgb_dir, self.input_rgb_images[idx])
        target_image_name = os.path.join(self.target_dir, self.target_images[idx])

        # if self.input_rgb_images[idx].split('_')[-1] == self.target_images[idx].split('_')[-1]:
        
        #     input_rgb_image = Image.open(input_image_rgb_name) 
        #     target_image = Image.open(target_image_name)
        # else: 
        #     print("MISMATCHED INPUTS!!!")
        #     print(input_image_rgb_name)
        #     print(target_image_name)

        input_rgb_image = Image.open(input_image_rgb_name) 
        target_image = Image.open(target_image_name)


        if self.color_format=='ycbcr':
            input_rgb_image = input_rgb_image.convert('YCbCr')
            target_image = target_image.convert('YCbCr')

        # to do
        # change the colour zones to ycbcr

        if self.transform: #transform is converting image to tensor as well 
            seed = torch.randint(0, 2**32, (1,)).item()  # Generate a new random seed
            torch.manual_seed(seed) 
            input_rgb_image = self.transform(input_rgb_image) 

            torch.manual_seed(seed)
            target_image = self.transform(target_image)
        else: 
            print('NO TRANSFORMS HAVE BEEN GIVEN')
        return input_rgb_image, target_image 
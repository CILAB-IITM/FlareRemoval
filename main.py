from venv import logger
import torch
from PIL import Image
import numpy as np
import os
from RealESRGAN import RealESRGAN
from tqdm import tqdm
from glob import glob


class SROps:
    def __init__(self, model_name, input_path, output_path, device, res):
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.device = device
        self.res = res

        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        if self.model_name == 'RealESRGAN':
            self.SRGan()
        else:
            print('Invalid model name')



    def SRGan(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)

        val_imgs = glob(val_path + '/*')

        for val_img in tqdm(val_imgs):

            image = Image.open(val_img).convert('RGB')
            sr_image = model.predict(image)

            img_name = val_img.split('/')[-1]
            output_path = os.path.join(self.output_path,  img_name)
            sr_image.save(output_path)


        logger.info('Super resolution completed using the model: {}'.format(self.model_name))


if __name__ == '__main__':
    val_path = '/home/saiteja/flare_IITM_Research/datasets/val_input_2k/val_input_2k_bicubic'
    output_path = '/home/saiteja/flare_IITM_Research/datasets/val_input_2k/val_input_2k_SR'
    model_name = 'RealESRGAN'
    device = 'cuda'
    res = 4
    srops = SROps(model_name, val_path, output_path, device, res).run()
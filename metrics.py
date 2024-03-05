from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from glob import glob


from torchvision import transforms
import pandas as pd


class Metrics:
     
    def __init__(self):
        vgg_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
        alex_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
        self.loss_fn_vgg = vgg_lpips
        self.loss_fn_alex = alex_lpips

        self.ssim = StructuralSimilarityIndexMeasure()
        self.metric = PeakSignalNoiseRatio()


    def run(self, source_img_list, target_img_list):

        mean_PSNR = 0
        mean_SSIM = 0
        mean_LPIPS_VGG = 0
        mean_LPIPS_ALEX = 0


        df  = pd.DataFrame(columns=['psnr', 'ssim', 'lpips_vgg', 'lpips_alex', 'source_img', 'target_img'])


        for source_img_path, target_img_path in tqdm(zip(source_img_list, target_img_list), total=len(source_img_list)):
            source_img = Image.open(source_img_path)
            target_img = Image.open(target_img_path)

            current_PSNR = self.compare_psnr(source_img, target_img)
            current_SSIM = self.compare_ssim(source_img, target_img)
            current_LPIPS_VGG = self.compare_lpips_vgg(source_img, target_img )
            current_LPIPS_ALEX = self.compare_lpips_alex(source_img, target_img)

            # get the psnr, ssim to numpy
            current_PSNR = current_PSNR.cpu().detach().numpy()
            current_SSIM = current_SSIM.cpu().detach().numpy()


            mean_PSNR += current_PSNR
            mean_SSIM += current_SSIM

            

            mean_LPIPS_VGG += current_LPIPS_VGG
            mean_LPIPS_ALEX += current_LPIPS_ALEX

            # data_frame = data_frame.append_({'psnr': current_PSNR, 'ssim': current_SSIM, 
            #                                 'lpips_vgg': current_LPIPS_VGG, 'lpips_alex': current_LPIPS_ALEX, 
            #                                 'source_img': source_img, 'target_img': target_img}, ignore_index=True)


            df.loc[len(df)] = [current_PSNR, current_SSIM, current_LPIPS_VGG, current_LPIPS_ALEX, source_img_path, target_img_path]

        mean_PSNR /= len(source_img_list)
        mean_SSIM /= len(source_img_list)
        mean_LPIPS_VGG /= len(source_img_list)
        mean_LPIPS_ALEX /= len(source_img_list)


        info = {'mean_PSNR': mean_PSNR, 'mean_SSIM': mean_SSIM, 'mean_LPIPS_VGG': mean_LPIPS_VGG, 'mean_LPIPS_ALEX': mean_LPIPS_ALEX}

        return info, df

    def compare_lpips_vgg(self, img1, img2):
        
        transform_ = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img1_tensor = transform_(img1).unsqueeze(0)
        img2_tensor = transform_(img2).unsqueeze(0)

        output_lpips = self.loss_fn_vgg(img1_tensor.cuda(), img2_tensor.cuda())
        return output_lpips.cpu().detach().numpy()


    def compare_lpips_alex(self, img1, img2):
        to_tensor=ToTensor()
        img1_tensor = to_tensor(img1).unsqueeze(0)
        img2_tensor = to_tensor(img2).unsqueeze(0)
        output_lpips = self.loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
        return output_lpips.cpu().detach().numpy()


    def compare_ssim(self, img1, img2):
        img1_tensor = ToTensor()(img1).unsqueeze(0)
        img2_tensor = ToTensor()(img2).unsqueeze(0)
        return self.ssim(img1_tensor, img2_tensor)

    def compare_psnr(self, img1, img2):
        img1_tensor = ToTensor()(img1).unsqueeze(0)
        img2_tensor = ToTensor()(img2).unsqueeze(0)
        return self.metric(img1_tensor, img2_tensor)


      



if __name__ == '__main__':
        

        import time

        og_list = glob('/home/cilab/teja/FlareRemoval/datasets/input/*')
        pred_list = glob('/home/cilab/teja/FlareRemoval/datasets/gt/*')

        og_list = sorted(og_list)
        pred_list = sorted(pred_list)

        og_list = og_list[:10]
        pred_list = pred_list[:10]

        unq_tag = time.time()


        metrics = Metrics()
        info, df = metrics.run(og_list, pred_list)

        # make a new directory
        import os
        os.makedirs(f'./results/{unq_tag}')
        df.to_csv(f'./results/{unq_tag}/results.csv')   

        # write the dict to a json file
        import json
        with open(f'./results/{unq_tag}/results.json', 'w') as f:
            json.dump(info, f)



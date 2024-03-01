import toml
import os
from scripts.patches.patches import make_patches  
from scripts.SuperResolution.super_res import super_res_main  
from tqdm import tqdm
from scripts.patches.patches import concat_patches 
import cv2

from scripts.Evaluate.evaluate import calculate_metrics


class FlareRemoval:
    def __init__(self,configs):
        self.configs  = configs

    def run(self):
        processes = configs['process']


        process_dict = {
            'model_inference': self.model_inference,
            'make_patches': self.make_patches,
            'super_res': self.super_res,
            'combine_patches': self.combine_patches,
            'resize': self.resize,
            'evaluate': self.evaluate,
        }

        for process in processes:
            func = process_dict[process]
            func(self.configs[process])

        return
    

    def model_inference(self, config):

        # current support for ufomer alone
        # python test_large.py --input dataset/Flare7Kpp/test_data/real/input 
        # --output result/test_real/flare7kpp/ 
        # --model_path experiments/flare7kpp/net_g_last.pth 
        # --flare7kpp

        script_path = 'scripts/Flare7K/test_large.py'
        env_path = config['env_path']
        input_path = config['input_path']
        output_path = config['output_path']
        model_path = config['model_path']
        # cmd1 = f'source {env_path} && '
        cmd2 = f'python {script_path} --input {input_path} --output {output_path} --model_path {model_path} --flare7kpp'
        print(cmd2)
        os.system(cmd2)
        return True
    


    def make_patches(self, configs):
        # input_path = '/home/saiteja/Desktop/ntire/Flare7K/output/blend'
        # output_path= 'output/patches'
        # patch_size = 256

        input_path = configs['input_path']
        output_path = configs['output_path']
        patch_size = configs['patch_size']
        make_patches(input_path, output_path, patch_size)


    def super_res(self, configs):
        input_path = configs['input_path'] 
        output_path = configs['output_path']
        super_res_main(input_path, output_path)

    def combine_patches(self, configs):
        input_path = configs['input_path']
        output_path = configs['output_path']
        # superrespath = '/home/saiteja/flare_IITM_Research/ImageSuperResolution/FlareRemoval/output_super_res'
        # concate_patches_path = 'output/concat'
        folds = os.listdir(input_path)
        for fol in tqdm(folds):
            # print("combining patches for the folder", fol)
            input_path_fold = os.path.join(input_path, fol)
            output_path = os.path.join(output_path)
            # concat_imgs(input_path=input_path, output_path, patch_size)
            concat_patches(input_path_fold, output_path)


    def resize(self, configs):
        og_imgs_path = configs['og_imgs_path']
        model_op_path = configs['model_op_path']
        output_path  = configs['output_path']

        os.makedirs(output_path, exist_ok=True)

        og_imgs = os.listdir(og_imgs_path)

        for img in tqdm(og_imgs):
            og_img = os.path.join(og_imgs_path, img)
            model_img = os.path.join(model_op_path, img)

            op_img = cv2.imread(og_img)
            model_img = cv2.imread(model_img)

            model_img = cv2.resize(model_img, (op_img.shape[1], op_img.shape[0]))
            cv2.imwrite(os.path.join(output_path, img), model_img)


    def evaluate(self, configs):
        # model_op_path = configs['model_op_path']
        # gt = configs['gt']
        configs['mask'] = None
        configs['input'] = configs['model_op_path']

        # cmd = 'python scripts/'
        calculate_metrics(configs)



if __name__ == '__main__':
    configs = toml.load('config.toml')
    FR = FlareRemoval(configs)
    FR.run()

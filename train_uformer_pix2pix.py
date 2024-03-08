import sys

from scripts.NAFNet.basicsr.utils.img_util import padding
sys.path.append('scripts/simplified_pix2pixHD')
sys.path.append('scripts/Flare7K')


import time


import numpy as np
import cv2
import torch
import losses
from functions import show_tensor
from discriminator import define_D
from replay_pool import ReplayPool
from torchvision import transforms
from random import uniform, randint
from glob import glob
import os
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
# from data_loader import Flare_Image_Loader
from scripts.models.Uformer import Uformer
from scripts.data_operations.uformer_dataset import Image_Dataset
import os
import wandb

import torch.nn as nn

# import lpips



from moving_average import moving_average
from tqdm import tqdm
from generator import define_G

api = '4b3b95fc9320ec524f3836b72046de4c1f343a4c'
# wandb.init(project="FlareRemoval")




		# transform_final = [
		# 	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		# ]

		# transform_final=transforms.Compose(transform_final)

		# if self.mask_type==None:
		# 	# return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),gamma
		
		# 	# m_img = PIL.from_array(adjust_gamma_reverse(merge_img))
		# 	# b_img = PIL.from_array(adjust_gamma_reverse(base_img))

		# 	m_img = adjust_gamma_reverse(merge_img)
		# 	# convert to PUIL image
		# 	# m_img = PIL.Image.fromarray(m_img.permute(1,2,0))

		# 	b_img = adjust_gamma_reverse(base_img)
		# 	# convert to PUIL image
		# 	# b_img = PIL.Image.fromarray(b_img.permute(1,2,0))


		# 	m_img = transform_final(adjust_gamma_reverse(merge_img))
		# 	b_img = transform_final(adjust_gamma_reverse(base_img))

		# 	return m_img, b_img


















replay_pool = ReplayPool(10)

def test(epoch, iteration, generator_ema, test_loader, images_output_dir, device):
    os.makedirs(images_output_dir, exist_ok=True)
    with torch.no_grad():
        data, target = next(iter(test_loader)) 
        data = data.to(device)
        generator_ema.eval()
        out = generator_ema(data)
        out = out[:, :3, :, :]
        generator_ema.train()
        matrix = []
        pairs = torch.cat([data, out, target.to(device)], -1)
        for idx in range(data.shape[0]):
            # img = 255*(pairs[idx] + 1)/2
            img = pairs[idx] * 255
            img = img.cpu().permute(1, 2, 0).clip(0, 255).numpy().astype(np.uint8)


            matrix.append(img)
        matrix = np.vstack(matrix)


        wandb.log({"image": [wandb.Image(matrix, caption=f"epoch {epoch}, iteration {iteration}")]})

        matrix = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
        # log the image
        out_file = os.path.join(images_output_dir, f"{epoch}_{iteration}.jpg")
        cv2.imwrite(out_file, matrix)

    
    # loss_fn_alex = lpips.LPIPS(net='alex')
    # loss_fn_vgg = lpips.LPIPS(net='vgg')
    # lpips_alex = 0
    # lpips_vgg = 0

    # with torch.no_grad():
    #     # run the ssim on all the images

    #     avg_ssim = 0
    #     for data, target in test_loader:
    #         data = data.to(device)
    #         target = target.to(device)
    #         out = generator_ema(data)

    #         pairs = torch.cat([data, out, target.to(device)], -1)
    #         for idx in range(data.shape[0]):
    #             img = 255*(pairs[idx] + 1)/2
    #             img = img.cpu().permute(1, 2, 0).clip(0, 255).numpy().astype(np.uint8)
    #             matrix.append(img)


    #         ssim_val = ssim(out, target, data_range=1, reduction="mean")
    #         avg_ssim += ssim_val
    #         lpips_alex += loss_fn_alex(out, target)
    #         lpips_vgg += loss_fn_vgg(out, target)

    #     lpips_alex = lpips_alex/len(test_loader)
    #     lpips_vgg = lpips_vgg/len(test_loader)
    #     ssim_val = avg_ssim/len(test_loader)
    #     wandb.log({"ssim": ssim_val, "lpips_alex": lpips_alex, "lpips_vgg": lpips_vgg})
        







def make_generator(name = 'uformer'):
    device = torch.device("cuda")

    if name == 'pix2pixhd':
        gen = define_G(input_nc = 3, 
                    output_nc = 3, 
                    ngf = 64, netG = "global", 
                    norm = "instance", 
                    n_downsample_global = 3, n_blocks_global = 9, 
                    n_local_enhancers = 1, n_blocks_local = 3).to(device)
        
    # to be filled by jaikar
    elif name == 'uformer':
        gen = Uformer(img_size=512,img_ch=3,output_ch= 6).to(device)
        # load the pretrained weights
        path = '/data/home/teja/diffusion_research/flareremoval/FlareRemoval/checkpoints/Flare/1709602120.118458/epoch_40_2024-03-05 09:35.pt'
        gen.load_state_dict(torch.load(path)['G'])
        # print(gen)
    return gen








# ckpt_file = "./checkpoints/facades/epoch_42_2021-06-23 00:44.pt"
# ckpt = torch.load(ckpt_file)
# generator.load_state_dict(ckpt["G"])
# generator_ema.load_state_dict(ckpt["G"])
# discriminator.load_state_dict(ckpt["D"])
# del ckpt




def process_loss(log, losses, weights =  None):
    loss = 0
    for k in losses:
        if k not in log:
            log[k] = 0
        log[k] += losses[k].item()
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + losses[k] * weights[k]
    return loss



# from scripts.Flare7K.basicsr.utils.flare_util import blend_light_source,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel



def calc_G_losses(data, target, generator,
                  criterionVGG, criterionGAN, criterionFeat, 
                  criterionl1,discriminator):
    
    # 3,512,512 -> 6,512,512
    fake = generator(data)

    # get the first 3 channels alone
    fake = fake[:, :3, :, :]
    l1_base = criterionl1(fake,target)
    loss_vgg  = criterionVGG(fake, target)
    pred_fake = discriminator(torch.cat([data, fake], axis=1))

    loss_adv = criterionGAN(pred_fake, 1)

    with torch.no_grad():
        pred_true = discriminator(torch.cat([data, target], axis=1))

    loss_adv_feat = 0
    adv_feats_count = 0        
    for d_fake_out, d_true_out in zip(pred_fake, pred_true):
        for l_fake, l_true in zip(d_fake_out[: -1], d_true_out[: -1]):
            loss_adv_feat = loss_adv_feat + criterionFeat(l_fake, l_true)
            adv_feats_count += 1
    loss_adv_feat = 1*(4/adv_feats_count)*loss_adv_feat
    lambda_feat = 10
    return {"G_vgg": loss_vgg, 
            "G_adv": loss_adv, 
            "G_adv_feat": lambda_feat*loss_adv_feat, 
            'G_l1': l1_base}








def calc_D_losses(data, target, generator, discriminator, criterionGAN):
    with torch.no_grad():
        gen_out = generator(data)

        # get the first 3 channels alone
        gen_out = gen_out[:, :3, :, :]
        fake = replay_pool.query({"input": data.detach(), "output": gen_out.detach()})
        
    pred_true = discriminator(torch.cat([data, target], axis=1))
    loss_true = criterionGAN(pred_true, 1)
    pred_fake = discriminator(torch.cat([fake["input"], fake["output"]], axis=1))
    loss_false = criterionGAN(pred_fake, 0)
    return {"D_true": loss_true, "D_false": loss_false}

def train(args, epoch,
          generator, generator_ema, discriminator,
          criterionVGG, criterionGAN, criterionFeat,
          G_optim, D_optim,
          train_loader, test_loader, images_output_dir, device, checkpoint_dir
          ):
    
    # epoch = args.epoch
    loss_weights = args.loss_weights
    N = 0
    log = {}
    pbar = tqdm(train_loader)

    wandb.log({"epoch": epoch})

    for data, target in pbar:
        with torch.no_grad():

            # 3,512,512
            data = data.to(device)

            # 6,512,512
            target = target.to(device)
        


        G_optim.zero_grad()

        # turn on the grad
        generator.requires_grad_(True)

        # turn on the grad for disc
        discriminator.requires_grad_(False)


        criterionl1 = nn.L1Loss()


        G_losses = calc_G_losses(data, target, generator, 
                                 criterionVGG, criterionGAN, criterionFeat,
                                 criterionl1,discriminator)
        

        G_loss = process_loss(log, G_losses, loss_weights)

        # log loss in wandb
        wandb.log({"G_loss": G_loss})
        # log all the loss
        for k in G_losses:
            wandb.log({f"G_{k}": G_losses[k]})

        G_loss.backward()
        del G_losses
        G_optim.step()
        moving_average(generator, generator_ema)
        
        D_optim.zero_grad()
        generator.requires_grad_(False)
        discriminator.requires_grad_(True)
        D_losses = calc_D_losses(data, target, generator, discriminator, criterionGAN)
        D_loss = process_loss(log, D_losses)
        # log loss in wandb
        wandb.log({"D_loss": D_loss})
        # log all the loss
        for k in D_losses:
            wandb.log({f"D_{k}": D_losses[k]})



        D_loss.backward()
        del D_losses
        D_optim.step()
        
        txt = ""
        N += 1
        if (N%100 == 0) or (N + 1 >= len(train_loader)):
            for i in range(2):
                test(epoch, N + i, generator_ema, test_loader, images_output_dir, device)
        
        
        for k in log:
            txt += f"{k}: {log[k]/N:.3e} "
        pbar.set_description(txt)
        

    for i in range(2):
        test(epoch, N, generator_ema, test_loader, images_output_dir, device)
        

    # if (N%1000 == 0) or (N + 1 >= len(train_loader)):
    if epoch % 10 == 0:
        import datetime
        out_file = f"epoch_{epoch}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.pt"
        out_file = os.path.join(checkpoint_dir, out_file)
        torch.save({"G": generator_ema.state_dict(), "D": discriminator.state_dict()}, out_file)
        print(f"Saved to {out_file}")


def runtrain(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        # print('===========================')
        # print(config)

        device = torch.device("cuda")

        generator = make_generator()
        # make a copy of generator
        generator_ema=make_generator()
        
        with torch.no_grad():
            for (gp, ep) in zip(generator.parameters(), generator_ema.parameters()):
                ep.data = gp.data.detach()

        discriminator = define_D(input_nc = 3 + 3, ndf = 64, 
                                n_layers_D = 4, num_D = 3, 
                                norm="instance", getIntermFeat=True).to(device)

        # loss functions
        criterionGAN = losses.GanLoss(use_lsgan=True).to(device)
        criterionFeat = torch.nn.L1Loss().to(device)
        criterionVGG = losses.VGGLoss().to(device)

        # optimizers
        G_optim = torch.optim.AdamW(generator.parameters(), config.lr)
        D_optim = torch.optim.AdamW(discriminator.parameters(), config.lr)

        transforms_base = transforms.Compose([
                                                    transforms.RandomCrop((800,800)),
                                                    # random rotation to 5 degrees

                                                    # random shear to 5 degrees
                                                    # transforms.RandomAffine(degrees=5, shear=5),

                                                    # resize
                                                    transforms.Resize((512, 512)),
                                                    transforms.RandomRotation(5),

                                                    # transforms.RandomResizedCrop(size=(512, 512)),
                                                    transforms.RandomHorizontalFlip(p=0.5),

                                                    # vertical flip
                                                    transforms.RandomVerticalFlip(p=0.5),

                                                    transforms.ToTensor(),
                                                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                ])
        

        dataset = Image_Dataset(config.dataset_path , color_format = config.color_format, transform=transforms_base)

        train_size = int(0.85*len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size =config.batch_size, shuffle = False )

        # train_loader = torch.utils.data.DataLoader(train_loader, config.batch_size, num_workers=4, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(val_loader, config.batch_size, num_workers=4, shuffle=True)

        # save the first few images in the batch
        # data, target = next(iter(train_loader))
        # show_tensor(data[0])
        # show_tensor(target[0])

        checkpoint_dir = "./checkpoints/Flare/" + str(time.time())
        images_output_dir = os.path.join(checkpoint_dir, "images")
        if not os.path.exists(images_output_dir):
            os.makedirs(images_output_dir)

        for epoch in range(config.epochs):
            generator.train()
            discriminator.train()
            train(config, epoch, generator, generator_ema, 
                  discriminator, criterionVGG, criterionGAN, 
                  criterionFeat, G_optim, D_optim, train_loader,
                    val_loader, images_output_dir, 
                    device, checkpoint_dir)


def Synthetic_data_loader(transform_base, transform_flare):
    flare_image_loader=Flare_Image_Loader('/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flickr24K',transform_base,transform_flare)
    flare_image_loader.load_scattering_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
    flare_image_loader.load_reflective_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Reflective_Flare')

    test_flare_image_loader=Flare_Image_Loader('/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/train_gt_2k',transform_base,transform_flare)
    test_flare_image_loader.load_scattering_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
    test_flare_image_loader.load_reflective_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Reflective_Flare')
    return flare_image_loader, test_flare_image_loader


sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'G_loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric


parameters_dict = {
    'loss_weights': {
        'values': [
            # {'G_adv': 0.1,'G_adv_feat': 0.1,'G_vgg': 1, 'G_l1': 10},
            # {'G_adv': 1,'G_adv_feat': 1,'G_vgg': 0.1, 'G_l1': 10},
            {'G_adv': 0.5,'G_adv_feat': 0.5,'G_vgg': 2, 'G_l1': 10},
            # {'G_adv': 0.3,'G_adv_feat': 0.3,'G_vgg': 1, 'G_l1': 10},
        ]
        }
    }

parameters_dict.update({
    'lr': {
        'values': [0.00001]}
    })

parameters_dict.update({
    'epochs': {
        'value': 100}
    })


parameters_dict.update({
    'batch_size': {
        'value': 2}
    })


# we are splitting the train and val in the train process
# the folders need to have gt and input subfolders inside them
parameters_dict.update({
    'dataset_path': {
        'value': 'datasets/'}
        # 'value': '/data/home/teja/diffusion_research/flareremoval/FlareRemoval/badperf_imgs/worst_train'}
        # 'value': 'dummy_dataset/'}

    },
    )

parameters_dict.update({
    'color_format': {
        'values': ['rgb']}
    },
    )

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="Pretrained_Paths")
wandb.agent(sweep_id, runtrain, count=1)
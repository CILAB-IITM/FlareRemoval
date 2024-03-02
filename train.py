import sys
sys.path.append('scripts/simplified_pix2pixHD')
sys.path.append('scripts/Flare7K')

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
from data_loader import Flare_Image_Loader
import os
import wandb

from piq import ssim
import lpips



from moving_average import moving_average
from tqdm import tqdm
from generator import define_G

api = '4b3b95fc9320ec524f3836b72046de4c1f343a4c'
# wandb.init(project="FlareRemoval")


replay_pool = ReplayPool(10)

def test(epoch, iteration, generator_ema, test_loader, images_output_dir, device):
    os.makedirs(images_output_dir, exist_ok=True)
    with torch.no_grad():
        data, target = next(iter(test_loader)) 
        data = data.to(device)
        generator_ema.eval()
        out = generator_ema(data)
        generator_ema.train()
        matrix = []
        pairs = torch.cat([data, out, target.to(device)], -1)
        for idx in range(data.shape[0]):
            img = 255*(pairs[idx] + 1)/2
            img = img.cpu().permute(1, 2, 0).clip(0, 255).numpy().astype(np.uint8)
            matrix.append(img)
        matrix = np.vstack(matrix)
        wandb.log({"image": [wandb.Image(matrix, caption=f"epoch {epoch}, iteration {iteration}")]})

        matrix = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
        # log the image
        out_file = os.path.join(images_output_dir, f"{epoch}_{iteration}.jpg")
        cv2.imwrite(out_file, matrix)

    
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    lpips_alex = 0
    lpips_vgg = 0

    with torch.no_grad():
        # run the ssim on all the images

        avg_ssim = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            out = generator_ema(data)
            ssim_val = ssim(out, target, data_range=1, reduction="mean")
            avg_ssim += ssim_val
            lpips_alex += loss_fn_alex(out, target)
            lpips_vgg += loss_fn_vgg(out, target)

        lpips_alex = lpips_alex/len(test_loader)
        lpips_vgg = lpips_vgg/len(test_loader)
        ssim_val = avg_ssim/len(test_loader)
        wandb.log({"ssim": ssim_val, "lpips_alex": lpips_alex, "lpips_vgg": lpips_vgg})
        

def make_generator():
    device = torch.device("cuda")
    gen = define_G(input_nc = 3, 
                output_nc = 3, 
                ngf = 64, netG = "global", 
                norm = "instance", 
                n_downsample_global = 3, n_blocks_global = 9, 
                n_local_enhancers = 1, n_blocks_local = 3).to(device)
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

def calc_G_losses(data, target, generator,
                  criterionVGG, criterionGAN, criterionFeat, discriminator):
    fake = generator(data)
    loss_vgg = 1*criterionVGG(fake, target)
    pred_fake = discriminator(torch.cat([data, fake], axis=1))
    loss_adv = 1*criterionGAN(pred_fake, 1)

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
    return {"G_vgg": loss_vgg, "G_adv": loss_adv, "G_adv_feat": lambda_feat*loss_adv_feat}

def calc_D_losses(data, target, generator, discriminator, criterionGAN):
    with torch.no_grad():
        gen_out = generator(data)
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
    batch_size = args.batch_size
    # loss_weights = {
    #     'gan': 1,
    #     'feat': 10,
    #     'VGG': 10
    # }
    # batch_size = 8
    N = 0
    log = {}
    pbar = tqdm(train_loader)
    for data, target in pbar:
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
        
        G_optim.zero_grad()
        generator.requires_grad_(True)
        discriminator.requires_grad_(False)
        G_losses = calc_G_losses(data, target, generator, criterionVGG, criterionGAN, criterionFeat, discriminator)
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
            for i in range(3):
                test(epoch, N + i, generator_ema, test_loader, images_output_dir, device)
        for k in log:
            txt += f"{k}: {log[k]/N:.3e} "
        pbar.set_description(txt)
        
        if (N%1000 == 0) or (N + 1 >= len(train_loader)):
            import datetime
            out_file = f"epoch_{epoch}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.pt"
            out_file = os.path.join(checkpoint_dir, out_file)
            # torch.save({"G": generator_ema.state_dict(), "D": discriminator.state_dict()}, out_file)
            print(f"Saved to {out_file}")

    # if epoch % 10 == 0:
    #     test(0, 0, make_generator(), test_loader, "./checkpoints/Flare/images", torch.device("cuda"))   % 10 == 0


# for epoch in range(0, 1):
#     train(epoch)

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
                                n_layers_D = 3, num_D = 2, 
                                norm="instance", getIntermFeat=True).to(device)

        # loss functions
        criterionGAN = losses.GANLoss(use_lsgan=True).to(device)
        criterionFeat = torch.nn.L1Loss().to(device)
        criterionVGG = losses.VGGLoss().to(device)

        # optimizers
        G_optim = torch.optim.AdamW(generator.parameters(), lr=1e-4)
        D_optim = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)

        transform_base=transforms.Compose([transforms.Resize((512,512)),
                                    transforms.RandomHorizontalFlip(),                            
                                    transforms.RandomVerticalFlip()
                                    ])

        transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(300/1440,300/1440),shear=(-20,20)),
                                    transforms.CenterCrop((512,512)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()
                                    ])

        flare_image_loader=Flare_Image_Loader('/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/data',transform_base,transform_flare)
        # flare_image_loader=Flare_Image_Loader('/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flickr24K',transform_base,transform_flare)

        flare_image_loader.load_scattering_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
        flare_image_loader.load_reflective_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Reflective_Flare')

        # test_flare_image_loader=Flare_Image_Loader('/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/train_gt_2k',transform_base,transform_flare)
        test_flare_image_loader=Flare_Image_Loader('/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/data',transform_base,transform_flare)
        test_flare_image_loader.load_scattering_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
        test_flare_image_loader.load_reflective_flare('Flare7K','/data/home/teja/diffusion_research/flareremoval/FlareRemoval/datasets/Flare7Kpp/Flare7K/Reflective_Flare')

        # flare_image_loader = flare_image_loader[:50]
        # test_flare_image_loader = test_flare_image_loader[:50]


        train_loader = torch.utils.data.DataLoader(flare_image_loader, config.batch_size, num_workers=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(flare_image_loader, config.batch_size, num_workers=4, shuffle=True)

        # save the first few images in the batch
        data, target = next(iter(train_loader))
        # show_tensor(data[0])
        # show_tensor(target[0])

        checkpoint_dir = "./checkpoints/Flare/"
        images_output_dir = os.path.join(checkpoint_dir, "images")
        if not os.path.exists(images_output_dir):
            os.makedirs(images_output_dir)

        for epoch in range(config.epochs):
            generator.train()
            discriminator.train()
            train(config, epoch, generator, generator_ema, 
                  discriminator, criterionVGG, criterionGAN, 
                  criterionFeat, G_optim, D_optim, train_loader,
                    test_loader, images_output_dir, 
                    device, checkpoint_dir)






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
            {'G_adv': 1,'G_adv_feat': 10,'G_vgg': 10},
            {'G_adv': 1,'G_adv_feat': 1,'G_vgg': 1},
            {'G_adv': 1,'G_adv_feat': 1,'G_vgg': 10},
        ]
        }
    }

parameters_dict.update({
    'epochs': {
        'value': 100}
    })


parameters_dict.update({
    'batch_size': {
        'value': 8}
    })

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
wandb.agent(sweep_id, runtrain, count=1)

import os
from glob import glob
from turtle import width
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2



def image_analysis(imgs_path):
    imgs = glob(os.path.join(imgs_path, '*.png'))

    sizes = []
    for img in imgs:
        image = Image.open(img).convert('RGB')
        width, height = image.size
        sizes.append((width, height))

    # print different heights and widths
    print(np.unique(sizes, axis=0))
    print('Total images: ', len(imgs))
    print('Total unique images: ', len(np.unique(sizes, axis=0)))



def check_patches(input_path, output_path):
    inp_imgs = glob(input_path + '/*')
    out_imgs = glob(output_path + '/*')
    loss  = 0
    for inpimg, opimg in tqdm(zip(inp_imgs, out_imgs), total=len(inp_imgs)):
        img1 = Image.open(inpimg)
        img2 = Image.open(opimg)
        img1 = np.array(img1)
        img2 = np.array(img2)
        loss += np.mean(np.abs(img1 - img2))
    print('Loss:', loss/len(inp_imgs))


from PIL import Image
import os
from glob import glob


def make_patches(input_path, output_path, patch_size = 512):
    imgs = glob(input_path + '/*')
    os.makedirs(output_path, exist_ok=True)

    for img in tqdm(imgs):
        image = Image.open(img).convert('RGB')
        width, height = image.size

        for i in range(0, width, patch_size):
            for j in range(0, height, patch_size):
                box = (i, j, i+patch_size, j+patch_size)
                patch = image.crop(box)
                patch.save(output_path + '/' + img.split('/')[-1].split('.')[0] + '_{}_{}.png'.format(j,i))






def concat_patches(input_path, output_path, patch_size = 512):
    imgs = glob(input_path + '/*')
    os.makedirs(output_path, exist_ok=True)

    all_unq_imgs = np.unique([img.split('/')[-1].split('_')[0] for img in imgs])

    sample_img = []

    for unq_img in all_unq_imgs:
        patches = glob(input_path + '/' + unq_img + '*.png')
        patches = sorted(patches, key=lambda x: int(x.split('_')[-2]))

        img_name_tag = []
        widht_tag = []
        height_tag = []

        for img in patches:
            # remove the extenstion .png
            img = img.split('.')[0]
            img_name_tag_, width_, height_ = img.split('/')[-1].split('_')
            img_name_tag.append(img_name_tag_)
            widht_tag.append(int(width_))
            height_tag.append(int(height_))

        # get the unq values
        img_name_tag = np.unique(img_name_tag)
        widht_tag = np.unique(widht_tag)
        height_tag = np.unique(height_tag)

        vertical_imgs  = []
        for height in height_tag:

            horizontal_imgs = []
            for widht in widht_tag:
                patch = Image.open(input_path + '/' + img_name_tag[0] + '_{}_{}.png'.format(widht,height)).convert('RGB')
                horizontal_imgs.append(patch)

            vertical_imgs.append(np.concatenate(horizontal_imgs, axis=0))
        
        # sample_img.append(np.concatenate(vertical_imgs, axis=0))

        # write the image
        img = np.concatenate(vertical_imgs, axis=1)
        img = Image.fromarray(img)
        img.save(output_path + '/' + unq_img + '.png')
        print(img_name_tag, widht_tag, height_tag)
    


if __name__ == '__main__':
    input_path = './op_test'
    output_path= 'output/patches'
    patch_size = 512
    make_patches(input_path, output_path, patch_size)

    # do a smart
    input_path = 'output/patches'
    output_path = 'output/concat'
    # concat_imgs(input_path=input_path, output_path, patch_size)
    # concat_patches(input_path, output_path, patch_size)


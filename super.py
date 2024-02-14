import shutil
from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests
import os

from glob import glob

imgs_fols = glob('output/patches/*')
# imgs = glob('results/*')


for img_fol in imgs_fols:

    fol_name = img_fol.split('/')[-1]

    imgs  = glob(img_fol + '/*')
    output_path = 'output_super_res/' + fol_name
    os.makedirs(output_path, exist_ok= True)

    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)      # scale 2, 3 and 4 models available
    model.to('cuda:0')

    from tqdm import tqdm
    for imgpath in tqdm(imgs):
        image = Image.open(imgpath)

        inputs = ImageLoader.load_image(image)
        preds = model(inputs)

        img_name = imgpath.split('/')[-1]
        output_img_path = os.path.join(output_path, img_name)
        ImageLoader.save_image(preds, output_img_path)                        # save the output 2x scaled image to `./scaled_2x.png`
        # ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')      # save an output comparing the super-image with a bicubic scaling
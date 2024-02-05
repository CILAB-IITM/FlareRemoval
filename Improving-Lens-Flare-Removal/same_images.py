import os
from glob import glob
import cv2
from tqdm import tqdm



def images_reshape_save(og_shapes, op_shapes, output_dir):

    # get all the images in the directory
    og_imgs = glob(og_shapes + '/*')
    op_imgs = glob(op_shapes + '/*')

    img_names = [os.path.basename(img) for img in og_imgs]

    # sort them
    og_imgs.sort()
    op_imgs.sort()

    img_extensions = ['.jpg', '.png', '.jpeg']

    # check if images are of same type
    for img in og_imgs:
        if not any(ext in img for ext in img_extensions):
            print('Image type not supported')
            return
        
    # create the op path
    os.makedirs(output_dir, exist_ok=True)
        

    for img in tqdm(img_names):
        og_img_path = os.path.join(og_shapes, img)
        op_img_path = os.path.join(op_shapes, img)

        # og shape
        og_img = cv2.imread(og_img_path)
        height, width, _ = og_img.shape

        op_img = cv2.imread(op_img_path)
        op_img = cv2.resize(op_img, (width, height))

        # save the images at the new path
        cv2.imwrite(os.path.join(output_dir, img), op_img)
    


if __name__ == '__main__':
    og_shapes = '/home/saiteja/flare_IITM_Research/datasets/submissions/val_input_2k/val_input_2k_bicubic'
    op_shapes =   '/home/saiteja/flare_IITM_Research/Improving-Lens-Flare-Removal/path/to/output/dir/output'
    output_dir = './reshaped_imgs'
    images_reshape_save(og_shapes, op_shapes, output_dir)

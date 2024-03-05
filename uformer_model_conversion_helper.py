folds = ['flikr_iitm_l1__0.5__vgg__0.5', 
         'flikr_iitm_l1__1.0__vgg__0.0', 
         'iitm_night_images_training__l1_0.5__vgg_0.5', 
         'iitm_night_images_training__l1_0.8__vgg_0.5', 
         'iitm_night_images_training__l1_1.0__vgg_0.0']

import os

# /data/tmp_teja/Flare7K/experiments/iitm_night_images_training__l1_0.5__vgg_0.5/models


for fold in folds:
    os.makedirs(f'model/{fold}', exist_ok=True)
    cmd = f'scp -r girish@10.24.6.143:/data/tmp_teja/Flare7K/experiments/{fold}/models model/{fold}'

    # print(cmd)
    os.system(cmd)
    # break


from glob import glob
models = glob('./model/*/models/*')

for mod in models:
    # os.rename(mod, mod.replace('models', 'model'))
    folder_name = mod.split('/')[-3]
    # change the model name to be folder name
    os.rename(mod, mod.replace('net_g_latest', folder_name))


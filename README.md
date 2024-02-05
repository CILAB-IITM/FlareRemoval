# FlareRemoval
This repo is about flare removal from images.


## Setting up
1. Clone the repo  and run the following commands for conda env
```bash
conda env create -f environment.yml
conda activate flare
```


2. Change the input and output path of the images in the execute.sh file.
```
bash execute.sh
```

```
python remove_flare.py --input_dir=/home/saiteja/flare_IITM_Research/datasets/submissions/val_input_2k/val_input_2k_bicubic   --out_dir=path/to/output/dir --model=Uformer    --batch_size=2    --ckpt=/home/saiteja/flare_IITM_Research/Improving-Lens-Flare-Removal/trained_model
```


from the above command, if generates 4 foldes,
1. input_images
2. output_images
3. output_blend
4. output_flare




3. The images of squared output which is different from the input images are shape.
```python
og_shapes = '/home/saiteja/flare_IITM_Research/datasets/submissions/val_input_2k/val_input_2k_bicubic'
op_shapes =   '/home/saiteja/flare_IITM_Research/Improving-Lens-Flare-Removal/path/to/output/dir/output'
output_dir = './reshaped_imgs'
```

```
python same_image.py
```
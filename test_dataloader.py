
import torch
from scripts.data_operations.uformer_dataset import Image_Dataset

class Config:
    def __init__(self) -> None:
        pass

config = Config()
config.dataset_path = 'datasets/'
config.color_format = 'rgb'
config.batch_size = 4



import torchvision.transforms as transforms


transforms_base = transforms.Compose([
                                            transforms.RandomCrop((512,512)),
                                            # transforms.RandomResizedCrop(size=(512, 512)),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                        ])


dataset = Image_Dataset(config.dataset_path , color_format = config.color_format, transform=transforms_base)

train_size = int(0.85*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size =config.batch_size, shuffle = False )


data,target  = next(iter(train_loader))

print(data.shape)
print(target.shape)
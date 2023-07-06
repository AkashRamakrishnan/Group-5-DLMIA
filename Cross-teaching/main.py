# import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.transform import resize
from dataset import *
from transformations import *
from model import *
from trainer import Trainer
from torchvision import transforms
from unet import UNet
from trainer import *
import torch.optim as optim
import monai

## Parameters
train_split = 0.8
train_bs = 1
val_bs = 1
test_bs = 16
in_channels = 1
n_classes = 4
data_transforms = transforms.Compose([transforms.Lambda(normalise), 
                                     transforms.Lambda(resize_image),
                                     transforms.Lambda(np_to_tensor)])
data_path = 'data/'

## Datasets
train_dataset = SegmentationDataSet(data_path, train=True, transform=data_transforms)
# add unlabeled train_dataset
test_dataset = SegmentationDataSet(data_path, train=False, transform=data_transforms)
train_size = int(train_split * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
print('Train Set: ', train_dataset.__len__())
print('Validation Set: ', val_dataset.__len__())
print('Test Set: ', test_dataset.__len__())

## Dataloader
dataloader_training = DataLoader(dataset=train_dataset,
                                 batch_size=train_bs,
                                 shuffle=True)
dataloader_validation = DataLoader(dataset=val_dataset,
                                 batch_size=val_bs,
                                 shuffle=True) 
dataloader_test = DataLoader(dataset=test_dataset,
                                 batch_size=test_bs,
                                 shuffle=True)

######### TO DO
# dataloader_unlabeled = DataLoader(...)

## Build Model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
# model = UNet3D(in_channels=in_channels, n_classes=n_classes).to(device)
unet_model = UNet(in_channels=1,
                  out_channels=4,
                  n_blocks=3,
                  start_filters=32,
                  activation='relu',
                  normalization='batch',
                  conv_mode='same',
                  dim=3).to(device)

unet_criterion = torch.nn.CrossEntropyLoss()
unet_optimizer = torch.optim.SGD(unet_model.parameters(), lr=0.01)
unet_scheduler = optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', patience=2, factor=0.5, verbose=True)

swin_model = monai.networks.nets.SwinUNETR(image_size=(256,256,40),
                                           in_channels=1,
                                           out_channels=4).to(device)

swin_criterion = torch.nn.CrossEntropyLoss()
swin_optimizer = torch.optim.SGD(swin_model.parameters(), lr=0.01)
swin_scheduler = optim.lr_scheduler.ReduceLROnPlateau(swin_optimizer, mode='min', patience=2, factor=0.5, verbose=True)

## train model
# trainer = Trainer(model=model,
#                   device=device,
#                   criterion=criterion,
#                   optimizer=optimizer,
#                   training_DataLoader=dataloader_training,
#                   validation_DataLoader=dataloader_test,
#                   lr_scheduler=None,
#                   epochs=10,
#                   epoch=0,
#                   notebook=False)

# training_losses, validation_losses, lr_rates = trainer.run_trainer()

train(unet_model=unet_model,
      swin_model=swin_model, 
      train_loader=dataloader_training,
      val_loader=dataloader_validation,
      unlabeled_loader=dataloader_unlabeled,
      unet_criterion=unet_criterion,
      unet_optimizer=unet_optimizer,
      unet_scheduler=unet_scheduler,
      swin_criterion=swin_criterion,
      swin_optimizer=swin_optimizer,
      swin_scheduler=swin_scheduler,
      device=device)

test(model=unet_model,
     test_loader = dataloader_test,
     criterion=unet_criterion, 
     device=device)
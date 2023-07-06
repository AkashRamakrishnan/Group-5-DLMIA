from monai.data import Dataset, DataLoader
from glob import glob
import os
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    Orientationd,
    RandAffined,
    RandRotated,
    RandGaussianNoised,
    NormalizeIntensity,
)
import random
### Libraries
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torch.optim import lr_scheduler

data_dir = '/content/drive/MyDrive/data/new_data'
train_images = sorted(glob(os.path.join(data_dir, 'train', 'images', '*.nii.gz')))
train_labels = sorted(glob(os.path.join(data_dir, 'train', 'targets', '*.nii.gz')))

test_images = sorted(glob(os.path.join(data_dir, 'test', 'images', '*.nii.gz')))
test_labels = sorted(glob(os.path.join(data_dir, 'test', 'targets', '*.nii.gz')))

data_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]

# random.shuffle(data_files)
val_split = 0.2
data_length = len(data_files)
val_length = int(val_split*data_length)
train_length = data_length - val_length
train_files = data_files[:train_length]
val_files = data_files[train_length:]

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # Resized(keys=["image", "label"], spatial_size=[256,256,10]),
        # NormalizeIntensity(keys="image"),

        # AddChanneld(keys=["image", "label"]),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
        # RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # Resized(keys=["image", "label"], spatial_size=[10,256,256]),
        # NormalizeIntensity(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
        ToTensord(keys=["image", "label"]),
    ]
)
train_set = Dataset(data=train_files, transform=train_transforms)
# train_set = Dataset(data=train_files)
model = UNet3D(num_classes=4, in_channels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print("device:", device)
model.to(device)
base_lr = 5e-3
## Define optimizer
optimizer = optim.SGD(model.parameters(), lr=base_lr,
                      momentum=0.9, weight_decay=0.0001)

## define loss function
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(4)

val_set = Dataset(data=val_files, transform=test_transforms)
test_set = Dataset(data=test_transforms, transform=test_transforms)

def plot_loss(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(f'loss_plot.png')
    plt.close() 

# Training loop
patience = 5
num_epochs = 100
total_iterations = len(train_dataloader)
# model.load_state_dict(torch.load('/content/drive/MyDrive/data/final_model.pt'))


def vadidation(val_dataloader,model, num_classes, set ):
  model.eval()
  val_loss = 0.0
  val_correct = 0
  total_val_samples = 0
  dice_score = 0
  with torch.no_grad():
      for data in val_dataloader:
          inputs = data['image'].to(device)
          labels = data['label'].to(device)
          inputs = inputs.unsqueeze(1)

          outputs = model(inputs)
          outputs_soft = torch.softmax(outputs, dim=1)
          loss_dice = dice_loss(outputs_soft, labels.unsqueeze(1))
          loss = loss_dice
          # val_loss += loss.item() * inputs.size(0)
          val_loss += loss.item()

          ##Calculating the dice score
          labels = one_hot_encoders(labels.unsqueeze(1), num_classes)
          outputs_soft = np.where(outputs_soft.cpu()> 0.5, 1, 0)

          dice_score += dc(np.array(outputs_soft), np.array(labels.cpu()))
  if (set =="validation"):
    print(f'Epoch {epoch + 1}/{num_epochs} \t Validation Loss: {val_loss / len(val_dataloader):.4f}')
    print(f'\n Dice_score for validation : {dice_score / len(val_dataloader):.4f}')
    return val_loss / len(val_dataloader)
  else:
    print(f'\n Dice_score for train : {dice_score / len(val_dataloader):.4f}')

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # print("epoch number: ",epoch)
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_train_samples = 0
    iter_num = 0
    for i, data in enumerate(train_dataloader):
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        inputs = inputs.unsqueeze(1)  # Add an extra dimension
        # inputs = inputs.permute(0, 2, 1, 3, 4)

        # print("inputs.shape:",inputs.shape)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs_soft = torch.softmax(outputs, dim=1)

        # Compute loss
        # print(outputs_soft.shape,labels.unsqueeze(1).shape)
        loss_dice = dice_loss(outputs_soft, labels.unsqueeze(1))
        loss = (loss_dice)
        # train_loss += loss.item() * inputs.size(0)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## validation

        ## print the loss values for each iteration
        iter_num += 1
        sys.stdout.write('\r')
        sys.stdout.write(
            f'Epoch {epoch + 1}/{num_epochs} \t Iteration {iter_num}/{total_iterations} \t Training Loss: {train_loss / (i + 1):.4f}'
        )
        sys.stdout.flush()
        # vadidation(val_dataloader,model,num_classes = 4 )

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} ')
    train_losses.append(train_loss/len(train_dataloader))
    # Validation
    avg_val_loss = vadidation(val_dataloader,model, num_classes = 4, set= 'validation')
    val_losses.append(avg_val_loss)
    plot_loss(train_losses, val_losses)
    # Validation
    # vadidation(train_dataloader,model, num_classes = 4, set= 'train')
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        # torch.save(model.state_dict(), '/content/drive/MyDrive/data/augmentation_model.pth')
            
    if epoch - best_epoch >= patience:
        print(f'Early stopping! No improvement in validation loss for {patience} epochs.')
        print(f'Best loss at Epoch {best_epoch}')
        torch.save(model.state_dict(), '/content/drive/MyDrive/data/augmentation_model.pth')
        break

import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import os
from tqdm import tqdm
from skimage.transform import resize
import nibabel as nib

def normalise(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.float32)
    return image

def resize_image(image):
    new_depth = 10
    new_width = 256
    new_height = 256
    image = resize(image, (new_depth, new_width, new_height),anti_aliasing=True , order=5,mode='constant')
    return image

def reshape_mask(mask):
    # Specify the desired new shape
    new_depth = 10
    new_width = 256
    new_height = 256

    # Reshape the mask using zoom and nearest neighbor interpolation
    resized_mask = zoom(mask, (new_depth / mask.shape[0], new_width / mask.shape[1], new_height / mask.shape[2]), order=0)

    # Convert the resized_mask to integers to remove any interpolated values
    resized_mask = resized_mask.astype(np.int32)
    return resized_mask

data_path = '/content/drive/MyDrive/data/data'
new_path = '/content/drive/MyDrive/data/new_data'

train_images = os.path.join(data_path, 'train', 'images')
train_images_out = os.path.join(new_path, 'train', 'images')

train_labels = os.path.join(data_path, 'train', 'targets')
train_labels_out = os.path.join(new_path, 'train', 'targets')

test_images = os.path.join(data_path, 'test', 'images')
test_images_out = os.path.join(new_path, 'test', 'images')

test_lables = os.path.join(data_path, 'test', 'targets')
test_labels_out = os.path.join(new_path, 'test', 'targets')

for fname in tqdm(os.listdir(train_images)):
    iname = os.path.join(train_images, fname)
    img = sitk.ReadImage(iname)
    arr = sitk.GetArrayFromImage(img)
    tarname = os.path.join(train_labels, fname)
    tar = sitk.ReadImage(tarname)
    tar_arr = sitk.GetArrayFromImage(tar)
    arr = resize_image(arr)
    arr = normalise(arr)
    tar = reshape_mask(tar_arr)
    # sitk.WriteImage(os.path.join(train_images_out, fname), arr)
    # sitk.WriteImage(os.path.join(train_labels_out, fname), tar)
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    print('File shape', img.shape)
    nib.save(img, os.path.join(train_images_out, fname))
    tar_img = nib.Nifti1Image(tar, affine=np.eye(4))
    nib.save(tar_img, os.path.join(train_labels_out, fname))



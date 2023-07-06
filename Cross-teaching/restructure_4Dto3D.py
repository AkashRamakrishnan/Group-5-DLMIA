import os
import glob
import numpy as np
import shutil
import SimpleITK as sitk
import nibabel as nib
from nibabel.testing import data_path

def restructure_4Dto3D(source_path, dest_path):
    # Training
    img4d_dir = os.path.join(source_path, 'train/4dvolumes')
    img3d_dest=os.path.join(dest_path, 'train/unlabeled3D')
    ci = 0
    if not os.path.exists(img3d_dest):
        os.makedirs(img3d_dest)
    for file_name in os.listdir(img4d_dir):
        source_file = os.path.join(img4d_dir, file_name)
        img4d_itk = sitk.ReadImage(source_file)
        img4d = sitk.GetArrayFromImage(img4d_itk)
        for i in range(img4d.shape[0]):
            img3d = img4d[i,:,:,:]
            img3d_nib = nib.Nifti1Image(img3d, img4d.affine)
            file_name_i = file_name[:-7] + '_frame_' + str(i) + file_name[-7:]
            destination_file = os.path.join(img3d_dest, file_name_i)
            nib.save(img3d_nib, os.path.join(file_name_i))
            ci +=1
    print('Transferred {ci} 3D images from 4D unlabeled data'.format(ci=ci))    
    
if __name__ == '__main__':
    source_path = "/home/jovyan/DLMIA-UNet-main/data/"
    dest_path = "/home/jovyan/DLMIA-UNet-main/data/"
    restructure_4Dto3D(source_path, dest_path)
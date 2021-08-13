### Original:
### Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
### BSD License. All rights reserved. 
### 
### Some sections:
### Copyright (c) 2019 Caroline Chan
### All rights reserved. 
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import os

def is_text_file(filename):
    return any(filename.endswith(extension) for extension in ['.txt', '.TXT'])

def make_text_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_text_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class AlignedPairedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### label maps    
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')              
        self.label_paths = sorted(make_dataset(self.dir_label))

        ### real images
        if opt.isTrain:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')  
            self.image_paths = sorted(make_dataset(self.dir_image))

    def __getitem__(self, index):        
        ### label maps
        paths = self.label_paths
        label_path = paths[index]              
        label = Image.open(label_path).convert('RGB')        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        original_label_path = label_path

        image_tensor = next_label = next_image = face_tensor = 0
        ### real images 
        if self.opt.isTrain:
            image_path = self.image_paths[index]   
            image = Image.open(image_path).convert('RGB')    
            transform_image = get_transform(self.opt, params)     
            image_tensor = transform_image(image)

        has_next = index < len(self) - 1

        """ Load the next label, image pair """
        if has_next:

            paths = self.label_paths
            label_path = paths[index+1]              
            label = Image.open(label_path).convert('RGB')        
            params = get_params(self.opt, label.size)          
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            next_label = transform_label(label)
            
            if self.opt.isTrain:
                image_path = self.image_paths[index+1]   
                image = Image.open(image_path).convert('RGB')
                transform_image = get_transform(self.opt, params)      
                next_image = transform_image(image)

        input_dict = {'label': label_tensor, 'image': image_tensor, 
                      'path': original_label_path, 'face_coords': face_tensor,
                      'next_label': next_label, 'next_image': next_image }
        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return 'AlignedPairedDataset'

### Allows building a dataset that is indexed by face coordinates to accomodate
### the case where not every frame has a face
class AlignedPairedFaceDataset(BaseDataset):
    def path_to_key(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### label maps    
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')              
        self.label_paths = sorted(make_dataset(self.dir_label))
        self.label_path_index = {self.path_to_key(p): i for i,p in enumerate(self.label_paths)}

        ### real images
        if opt.isTrain:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')  
            self.image_paths = sorted(make_dataset(self.dir_image))
            self.image_path_index = {self.path_to_key(p): i for i,p in enumerate(self.image_paths)}

        ### load face bounding box coordinates size 128x128
        self.dir_facecoords = os.path.join(opt.dataroot, opt.phase + '_facecoords')
        self.facecoord_paths = sorted(make_text_dataset(self.dir_facecoords))

    def __getitem__(self, index):        
        face_path = self.facecoord_paths[index]
        key = self.path_to_key(face_path)
        lindex = self.label_path_index[key]

        ### Face
        face = open(face_path, "r").read()
        face_tensor = torch.IntTensor(list([int(coord_str) for coord_str in face.split()]))

        ### label maps
        label_path = self.label_paths[lindex]              
        label = Image.open(label_path).convert('RGB')        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        original_label_path = label_path

        image_tensor = next_label = next_image = 0
        ### real images 
        if self.opt.isTrain:
            iindex = self.image_path_index[key]
            image_path = self.image_paths[iindex]   
            image = Image.open(image_path).convert('RGB')    
            transform_image = get_transform(self.opt, params)     
            image_tensor = transform_image(image)

        has_next = index < len(self) - 1

        """ Load the next label, image pair """
        if has_next:
            face_path = self.facecoord_paths[index+1]
            key = self.path_to_key(face_path)
            lindex = self.label_path_index[key]

            label_path = self.label_paths[lindex]
            label = Image.open(label_path).convert('RGB')        
            params = get_params(self.opt, label.size)          
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            next_label = transform_label(label)
            
            if self.opt.isTrain:
                iindex = self.image_path_index[key]
                image_path = self.image_paths[iindex]
                image = Image.open(image_path).convert('RGB')
                transform_image = get_transform(self.opt, params)      
                next_image = transform_image(image)


        input_dict = {'label': label_tensor, 'image': image_tensor, 
                      'path': original_label_path, 'face_coords': face_tensor,
                      'next_label': next_label, 'next_image': next_image }
        return input_dict

    def __len__(self):
        return len(self.facecoord_paths)

    def name(self):
        return 'AlignedPairedFaceDataset'

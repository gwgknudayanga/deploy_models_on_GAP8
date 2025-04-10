import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from typing import Callable, List, Tuple, Any, Dict
from torch.utils.data import Dataset

# import slayer from lava-dl
#import lava.lib.dl.slayer as slayer

import IPython.display as display
from matplotlib import animation
import cv2
from .input_transforms import augment,letterbox,letterbox2,vflip,hflip,rotate90antclk,rotate90clk,random_affine
from .visualize_data import dump_image_with_labels,make_dvs_frame
import cv2

data_visualize_output_path = "/home/udayanga/Udaya_Research_stuff/2024_GAP8_work/yolov3_ann_model_images/visualize_data"

def coco2bbox(anns):
    
    anns[:,3] += anns[:,1]
    anns[:,4] += anns[:,2]
    
def yolo2bbox(ann,img_width,img_height):

    x1 = (ann[:,1] - (ann[:,3] / 2)) * img_width
    y1 = (ann[:,2] - (ann[:,4] / 2)) * img_height
    x2 = (ann[:,1] + (ann[:,3] / 2)) * img_width
    y2 = (ann[:,2] + (ann[:,4] / 2)) * img_height
    ann[:,1] = x1
    ann[:,2] = y1
    ann[:,3] = x2
    ann[:,4] = y2

def round_to_nearest_1e3(x):
    return int(round(x / 1e3) * 1e3)
 
class evCIVIL(Dataset):

    def __init__(self,root: str = '.',csv_file_name: str = '.',train: bool = False,augment: bool = False,param_dict = {"TSteps" : 10, "tbins" : 2,"quantized_h" : 260 ,"quantized_w" : 346},target_spatial_size = (320,320))-> None:
        super().__init__()
        
        csv_file_path = os.path.join(root,csv_file_name)
        self.image_data_root = root

        with open(csv_file_path,"r") as file:
            self.img_files = file.readlines()

        self.param_dict = param_dict
        self.augment = augment
        self.target_spatial_size = target_spatial_size
        self.frame_based = True
        
    def __getitem__(self, index: int):

        local_event_npz_path = self.img_files[index % len(self.img_files)].rstrip()
        event_npz_path = os.path.join(self.image_data_root,local_event_npz_path)
        #print("event npz path ",event_npz_path)
        data_file = np.load(event_npz_path)

        frame_img = data_file["frame_img"]
        ann_array = data_file["ann_array"]

        #print("events shape ",events.shape)
        #print("ann_array shape ",ann_array.shape)
        coco2bbox(ann_array)
        #Now the anns are in bbox format
        """if self.augment:
            #if random.random() < 0.8:
            events,ann_array = augment(events,ann_array)"""

        #event_cube = get_event_cube(events,self.param_dict)
        #event_cube = event_cube.to_dense().permute(0,3,2,1)
        #Now the tensor is in TCHW format

        #get_histogram_sequence()

        frame_img = frame_img[np.newaxis,:,:]
        im,ann_array,ratio_tuple, pad_tuple = letterbox(torch.from_numpy(frame_img),ann_array,new_shape = self.target_spatial_size[0])

        if self.augment:
            im = im.permute(1,2,0) #H,W,C
            im,ann_array= random_affine(im.numpy(),
                    ann_array,
                    degrees=0.1, #0.0
                    translate=0.1,
                    scale=0.5,
                    shear=0.2,   #0.0
                    new_shape=(self.target_spatial_size[0],self.target_spatial_size[1]))
            
            im = im[:,:,np.newaxis]
            im = im.transpose(2,0,1) #C,H,W

            im = torch.from_numpy(im) 
        
        #Now the im should be in CHW format.

        im = im.repeat(1,1,1,1).float()
        #Now the im should be in TCHW format.

        if self.augment:
            if random.random() < 0.6:
                im,ann_array = vflip(im,ann_array,self.target_spatial_size[0] - 1)
            if random.random() < 0.5:
                im,ann_array = hflip(im,ann_array,self.target_spatial_size[1] - 1)
            if random.random() < 0.5:
                im,ann_array = rotate90antclk(im,ann_array,self.target_spatial_size[0])
            if random.random() < 0.5:
                im,ann_array = rotate90clk(im,ann_array,self.target_spatial_size[1])
        
        #In this case the ann array should be in bbox format
        ann_array = ann_array.reshape(-1,5)
        temp = ann_array[:,1:].copy()
        temp[:,0] = (ann_array[:,1] + ann_array[:,3]) / 2
        temp[:,1] = (ann_array[:,2] + ann_array[:,4]) / 2
        temp[:,2] = (ann_array[:,3] - ann_array[:,1])
        temp[:,3] = (ann_array[:,4] - ann_array[:,2])
        
        temp[:,0] /= self.target_spatial_size[1]
        temp[:,2] /= self.target_spatial_size[1]
        temp[:,1] /= self.target_spatial_size[0]
        temp[:,3] /= self.target_spatial_size[0]

        ann_array[:,1:] = temp
        ann_array = ann_array[:,[1,2,3,4,0]]
        conf_column = np.ones(ann_array.shape[0])
        ann_array = np.insert(ann_array, 4, conf_column, axis=1)
        #print("target spatial size ",self.target_spatial_size)
        #dump_image_with_labels(im.squeeze(),ann_array,self.target_spatial_size,data_visualize_output_path,(local_event_npz_path.rsplit(".",1)[0]).rsplit("/",1)[1],create_histo_frames = False)
        im = im.permute(1,2,3,0)
        im = (im/128.0) - 1
        #Now im is in CHWT format
        return im,[torch.tensor(ann_array, dtype=torch.float32)]

    
    def __len__(self) -> int :
        
        return len(self.img_files)
    

if __name__== "__main__":

    dataset_path = "/home/udayanga/latest_dataset"
    csv_file_name = "night_outdoor_and_daytime_train_files_image_based.txt"
    test_csv_file = "test_files_image_based.txt"
    dataset = evCIVIL(root=dataset_path,csv_file_name=test_csv_file,train=False,augment=False,param_dict=None,target_spatial_size = (240,320))
    
    im,ann = dataset.__getitem__(1400)

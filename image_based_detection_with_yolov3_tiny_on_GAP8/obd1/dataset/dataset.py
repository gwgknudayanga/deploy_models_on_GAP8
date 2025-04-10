import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from typing import Callable, List, Tuple, Any, Dict
from torch.utils.data import Dataset

# import slayer from lava-dl

import IPython.display as display
from matplotlib import animation
import cv2
from .input_transforms import augment,letterbox,vflip,hflip,rotate90antclk,rotate90clk
from .visualize_data import dump_image_with_labels,make_dvs_frame

data_visualize_output_path = "/media/atiye/Data/Udaya_Research_stuff/spikingjelly/yolov3/visualize_data_folder/"


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
 
def get_event_cube(events1,param_dict = {"Tsteps" : 7, "tbins" : 1,"quantized_h" : 260 ,"quantized_w" : 346} ):
    
    tbin = param_dict["tbins"]
    C = 2 * tbin
    quantized_h = param_dict["quantized_h"]
    quantized_w = param_dict["quantized_w"]
    T = param_dict["TSteps"]
    
    #sample = file_path #"/dtu/eumcaerotrain/data/latest_dataset/dset_1/npz_files_event_based/crack/crack_20/crack_20_7572871.npz"
    #data = np.load(file_path)

    events = events1 #data["events"]
    events[:,0] -= events[0,0]
    sample_size = int(round_to_nearest_1e3(events[:,0].max()))
    #print("sample size is ",sample_size)
    quantization_size = [np.ceil(sample_size / T), 1, 1]

    events = events[events[:,0] < sample_size,:]
    coords1 = events[:,:3]
    coords = torch.floor(coords1 / torch.tensor(quantization_size))
    coords[:,0] = coords[:,0].clamp(min = 0)
    coords[:,1] = coords[:,1].clamp(min = 0,max = quantized_w - 1)
    coords[:,2] = coords[:,2].clamp(min = 0, max = quantized_h - 1)
    tbin_size = quantization_size[0] / tbin
    tbin_coords = (events[:,0] % quantization_size[0]) // tbin_size
    tbin_feats = (2 * tbin_coords) + events[:,3]
    feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*tbin)

    """
    ##Start : New code for dense tensor
    dense_tensor = torch.zeros((T, quantized_w, quantized_h, 2 * tbin), dtype=torch.float32)
    # Vectorized operation to fill the dense tensor

    indices = coords.to(torch.int32)
    print("T max ",indices[:, 0].max())
    print("x max ",indices[:, 1].max())
    print("y max ",indices[:, 2].max())


    if indices[:, 0].max() > 6:
        print("hiiiii ",indices[:, 0].max(),"  ",coords[:,0].max())

    dense_tensor.index_put_((indices[:, 0], indices[:, 1], indices[:, 2]), feats.float(), accumulate=True)
    dense_tensor = dense_tensor.to(bool)
    dense_tensor = dense_tensor.to(torch.float32)
    ## End: New code for dense tensor"""
     
    sparse_tensor = torch.sparse_coo_tensor(
    coords.t().to(torch.int32),
    feats,
    size=(T,quantized_w,quantized_h,C),
    ).coalesce()
    sparse_tensor = sparse_tensor.to(bool)
    sparse_tensor = sparse_tensor.to(torch.float32)
    #return shape [T,w,h,C]
    #if not is_sparse:
    #    sparse_tensor.to_dense().permute(0,3,1,2)
    #return shape [T,w,h,C]     
    return sparse_tensor.to_dense()
    
class evCIVIL(Dataset):

    def __init__(self,root: str = '.',csv_file_name: str = '.',train: bool = False,augment: bool = False,
                 param_dict = {"TSteps" : 10, "tbins" : 2,"quantized_h" : 260 ,"quantized_w" : 346},target_spatial_size = 320)-> None:
        super().__init__()
        
        csv_file_path = os.path.join(root,csv_file_name)
        self.image_data_root = root

        with open(csv_file_path,"r") as file:
            self.img_files = file.readlines()

        self.param_dict = param_dict
        self.augment = augment
        self.target_spatial_size = target_spatial_size

    def __getitem__(self, index: int):

        local_event_npz_path = self.img_files[index % len(self.img_files)].rstrip()
        event_npz_path = os.path.join(self.image_data_root,local_event_npz_path)
        #print("event npz path ",event_npz_path)
        data_file = np.load(event_npz_path)

        events = data_file["events"]
        ann_array = data_file["ann_array"]
        #print("events shape ",events.shape)
        #print("ann_array shape ",ann_array.shape)
        coco2bbox(ann_array)
        #Now the anns are in bbox format

        event_cube = get_event_cube(events,self.param_dict)
        #event_cube = torch.randn((7,346,260,2))
        event_cube = event_cube.permute(0,3,2,1) #event_cube.to_dense().permute(0,3,2,1)
        #Now the tensor is in TCHW format

        #get_histogram_sequence()

        """event_cube = make_dvs_frame(events, height=260,width=346,color=True, clip=None,forDisplay = False)

        event_cube = event_cube[np.newaxis,:,:,:]
        event_cube = np.repeat(event_cube,4,axis=0)
        event_cube = event_cube.transpose(0,3,1,2)
        event_cube = event_cube.astype(np.float32)
        event_cube = torch.from_numpy(event_cube)"""
        #Now the tensor is in TCHW format

        #print("event cube maxxx ",event_cube.max(),"    ",event_cube.min())

        event_cube,ann_array,ratio_tuple, pad_tuple = letterbox(event_cube,ann_array,new_shape = self.target_spatial_size)
        
        event_cube = event_cube.permute(1,2,3,0)
        #Now the tensor is in CHWT format

        #In this case the ann array should be in bbox format
        ann_array = ann_array.reshape(-1,5)
        temp = ann_array[:,1:].copy()
        temp[:,0] = (ann_array[:,1] + ann_array[:,3]) / 2
        temp[:,1] = (ann_array[:,2] + ann_array[:,4]) / 2
        temp[:,2] = (ann_array[:,3] - ann_array[:,1])
        temp[:,3] = (ann_array[:,4] - ann_array[:,2])
        temp /= self.target_spatial_size

        temp = np.clip(temp,0,0.999)

        ann_array[:,1:] = temp
        ann_array = ann_array[:,[1,2,3,4,0]]
        conf_column = np.ones(ann_array.shape[0])
        ann_array = np.insert(ann_array, 4, conf_column, axis=1)

        #add additional dimension to cope with time to compatible with architecture
        
        #dump_image_with_labels(event_cube.permute(3,1,2,0),ann_array,self.target_spatial_size,data_visualize_output_path,(local_event_npz_path.rsplit(".",1)[0]).rsplit("/",1)[1],create_histo_frames = True)
        
        #bbb = torch.zeros(event_cube.shape[:-1]).unsqueeze(-1)
        #print("shapeeeeeeeeeeeeeeeeee ",event_cube.shape)
        #ccc = torch.concatenate((event_cube,bbb),dim=-1)
        event_cube = event_cube.permute(2,1,0,3) #Now WHCT format
        event_cube = torch.concatenate((event_cube,torch.zeros(320,320,2,1)),dim=-1)
        #print("data cube shape vs index ",event_cube.shape," ",index)
        return event_cube.numpy(),index #torch.tensor(ann_array, dtype=torch.float32).numpy()
    
    def __getelement__(self, index: int):

        local_event_npz_path = self.img_files[index % len(self.img_files)].rstrip()
        event_npz_path = os.path.join(self.image_data_root,local_event_npz_path)
        #print("event npz path ",event_npz_path)
        data_file = np.load(event_npz_path)

        events = data_file["events"]
        ann_array = data_file["ann_array"]
        #print("events shape ",events.shape)
        #print("ann_array shape ",ann_array.shape)
        coco2bbox(ann_array)
        #Now the anns are in bbox format

        event_cube = get_event_cube(events,self.param_dict)
        #event_cube = torch.randn((7,346,260,2))
        event_cube = event_cube.permute(0,3,2,1) #event_cube.to_dense().permute(0,3,2,1)
        #Now the tensor is in TCHW format

        #get_histogram_sequence()

        """event_cube = make_dvs_frame(events, height=260,width=346,color=True, clip=None,forDisplay = False)

        event_cube = event_cube[np.newaxis,:,:,:]
        event_cube = np.repeat(event_cube,4,axis=0)
        event_cube = event_cube.transpose(0,3,1,2)
        event_cube = event_cube.astype(np.float32)
        event_cube = torch.from_numpy(event_cube)"""
        #Now the tensor is in TCHW format

        #print("event cube maxxx ",event_cube.max(),"    ",event_cube.min())

        event_cube,ann_array,ratio_tuple, pad_tuple = letterbox(event_cube,ann_array,new_shape = self.target_spatial_size)
        
        event_cube = event_cube.permute(1,2,3,0)
        #Now the tensor is in CHWT format

        #In this case the ann array should be in bbox format
        ann_array = ann_array.reshape(-1,5)
        temp = ann_array[:,1:].copy()
        temp[:,0] = (ann_array[:,1] + ann_array[:,3]) / 2
        temp[:,1] = (ann_array[:,2] + ann_array[:,4]) / 2
        temp[:,2] = (ann_array[:,3] - ann_array[:,1])
        temp[:,3] = (ann_array[:,4] - ann_array[:,2])
        temp /= self.target_spatial_size

        temp = np.clip(temp,0,0.999)

        ann_array[:,1:] = temp
        ann_array = ann_array[:,[1,2,3,4,0]]
        conf_column = np.ones(ann_array.shape[0])
        ann_array = np.insert(ann_array, 4, conf_column, axis=1)

        #add additional dimension to cope with time to compatible with architecture
        
        #dump_image_with_labels(event_cube.permute(3,1,2,0),ann_array,self.target_spatial_size,data_visualize_output_path,(local_event_npz_path.rsplit(".",1)[0]).rsplit("/",1)[1],create_histo_frames = True)
        
        #bbb = torch.zeros(event_cube.shape[:-1]).unsqueeze(-1)
        #print("shapeeeeeeeeeeeeeeeeee ",event_cube.shape)
        #ccc = torch.concatenate((event_cube,bbb),dim=-1)
        #event_cube = event_cube.permute(2,1,0,3) #Now WHCT format
        #event_cube = torch.concatenate((event_cube,torch.zeros(320,320,2,1)),dim=-1)
        return event_cube,torch.tensor(ann_array, dtype=torch.float32)
    
    
    def __len__(self) -> int :
        
        return len(self.img_files)

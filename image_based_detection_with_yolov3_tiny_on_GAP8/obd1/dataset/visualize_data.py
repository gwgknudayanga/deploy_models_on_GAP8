
import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
import torchvision.transforms as Tr
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id,img_array,isImgFrame,isvoxelgrid = False,target_size = (346,260)):

    #Image array should be (width,height) shape

    print("path_to_save ",modified_images_to_matchtest_path)
    modifier = "_evframe"
    if isImgFrame:
        modifier = "_frame"

    print("initial shape ",img_array.shape)
    relative_out_fname = str(file_id) + modifier + ".png"
    output_full_fname = os.path.join(modified_images_to_matchtest_path,relative_out_fname)

    print("relative_out_fname ",relative_out_fname)

    fig,ax = plt.subplots(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    
    fig.add_axes(ax)
    if isvoxelgrid:
        ax.imshow(img_array[0],cmap="gray",aspect="auto")
    else:
        ax.imshow(img_array,cmap="gray",aspect="auto")

    #plt.close()
    #fig.savefig(output_full_fname)
    fig.canvas.draw()             
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    print("width and height ",fig.canvas.get_width_height())
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    print("initial shape 2222 ",data.shape)
    plt.close()
    #print("data.shape",data)
    im = Image.fromarray(data)
    im = im.resize(target_size)
    print("output full path ",output_full_fname)
    im.save(output_full_fname)
    with open(output_full_fname, 'rb') as f:
        pass  # Just open and close the file
    return output_full_fname

def get_events_only_2D_tensor(dense_4D_tensor):
    #indices = torch.nonzero(dense_4D_tensor > 0, as_tuple=False)
    return torch.nonzero(dense_4D_tensor > 0, as_tuple=False)

def make_dvs_frame(events, height=None, width=None, color=True, clip=3,forDisplay = False):
    """Create a single frame.

    Mainly for visualization purposes

    # Arguments
    events : np.ndarray
        (t, x, y, p)
    x_pos : np.ndarray
        x positions
    """
    if height is None or width is None:
        height = events[:, 2].max()+1
        width = events[:, 1].max()+1

    histrange = [(0, v) for v in (height, width)]

    pol_on = (events[:, 3] == 1)
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            events[pol_on, 2], events[pol_on, 1],
            bins=(height, width), range=histrange)
    img_off, _, _ = np.histogram2d(
            events[pol_off, 2], events[pol_off, 1],
            bins=(height, width), range=histrange)

    on_non_zero_img = img_on.flatten()[img_on.flatten() > 0]
    on_mean_activation = np.mean(on_non_zero_img)
    off_non_zero_img = img_off.flatten()[img_off.flatten() > 0]
    off_mean_activation = np.mean(off_non_zero_img)

    # on clip
    if clip is None:
        on_std_activation = np.std(on_non_zero_img)
        img_on = np.clip(
            img_on, on_mean_activation-3*on_std_activation,
            on_mean_activation+3*on_std_activation)
    else:
        img_on = np.clip(
            img_on, -clip, clip)

    # off clip
    
    if clip is None:
        off_std_activation = np.std(off_non_zero_img)
        img_off = np.clip(
            img_off, off_mean_activation-3*off_std_activation,
            off_mean_activation+3*off_std_activation)
    else:
        img_off = np.clip(
            img_off, -clip, clip)

    if color:

        frame = np.zeros((height, width, 2))
        img_on /= img_on.max()
        frame[..., 0] = img_on
        """img_on -= img_on.min()
        img_on /= img_on.max()"""

        img_off /= img_off.max()
        frame[..., 1] = img_off
        """img_off -= img_off.min()
        img_off /= img_off.max()"""

        #print("absolute max and min = ",np.abs(frame).max())
        if forDisplay:
            third_channel = np.zeros((height,width,1))
            frame = np.concatenate((frame,third_channel),axis=2)

    else:
        frame = img_on - img_off
        #frame -= frame.min()
        #frame /= frame.max()
        frame /= np.abs(frame).max()

    return frame

def draw_labels_on_image(image_path,ann_array,format = "yolo"):
    
    #annotations are expected to be in bbox format
    
    num_of_anns = len(ann_array)
    image = cv2.imread(image_path)
    #print("image shape ",image.shape)
    if image is None:
        print("no image forrrrrrrrrrrrrr ")
        return
    
    h,w = image.shape[:2] 
    for idx in range(num_of_anns):
        annotation = ann_array[idx]
        class_label = int(annotation[0])
        if format == "bbox":
            x_min = int(annotation[1])
            y_min = int(annotation[2])
            x_max = int(annotation[3])    #+ x_min
            y_max = int(annotation[4])    #+ y_min
        elif format == "yolo":
            x_min = int((annotation[1] - annotation[3]/2) * w)
            y_min = int((annotation[2] - annotation[4]/2) * h)
            x_max = int((annotation[1] + annotation[3]/2) * w)    #+ x_min
            y_max = int((annotation[2] + annotation[4]/2) * h)    #+ y_min
        elif format == "normalized_bbox":
            x_min = int(annotation[1] * w)
            y_min = int(annotation[2] * h)
            x_max = int(annotation[3] * w)
            y_max = int(annotation[4] * h)
        elif format == "coco":
            x_min = int(annotation[1])
            y_min = int(annotation[2])
            x_max = int(annotation[3]) + x_min
            y_max = int(annotation[4]) + y_min

        #print(class_label," ",x_min," ",y_min," ",x_max," ",y_max)
        
        # Load the image
        top_left =  (x_min,y_min)
        bottom_right = (x_max,y_max)
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

        label_text = ""
        
        if class_label == 0:
            label_text = "crack"
        elif class_label == 1:
            label_text = "spalling"
            
        # Define the position and font settings for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (255, 0, 0)  # White color
        text_position = (top_left[0], top_left[1] - 10)

        cv2.putText(image, label_text, text_position, font, font_scale, font_color, font_thickness)
        
        # Display the image with the rectangle
    cv2.imwrite(image_path, image)


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


def dump_img_with_ann(image_output_folder,image_name,img_numpy_array,ann_array,isImgFrame = True,format="bbox"):
    # the ann array should be in coco format
    output_file_name = save_images_for_matchscore_calculation(image_output_folder,file_id = image_name,img_array = img_numpy_array,isImgFrame = isImgFrame)
    img_path = os.path.join(image_output_folder,output_file_name)
    draw_labels_on_image(img_path,ann_array,format)


def get_single_channel_hist_from_event_cube(event_cube):

    #dense_permuted = event_cube.to_dense().permute(0,3,2,1) # (T,C,h,w)

    pos = event_cube[:,0::2,:,:]
    neg = event_cube[:,1::2,:,:]

    pos_hist = torch.sum(pos,dim = 0)
    pos_hist = torch.sum(pos_hist,dim=0)

    neg_hist = torch.sum(neg,dim = 0)
    neg_hist = torch.sum(neg_hist,dim = 0)

    frame = np.clip(pos_hist,-3,3) - np.clip(neg_hist,-3,3)
    frame /= np.abs(frame).max()
    return frame


def dump_image_with_labels(event_cube,ann_array,target_spatial_size,output_path,index,create_histo_frames = True):

    desired_ann = ann_array[:, [0,1,2,3,5]]
    desired_ann[:,[0,1,2,3,4]] = desired_ann[:,[4,0,1,2,3]]

    dvs_frame = event_cube.numpy()
    img_full_name = save_images_for_matchscore_calculation(output_path,file_id = index,img_array = dvs_frame,isImgFrame = False,target_size=target_spatial_size)
    
    if img_full_name:
        draw_labels_on_image(img_full_name,desired_ann,format = "yolo")

"""if __name__ == "__main__":

    src = "/home/udayanga/Desktop/test_samples/nonAccept046055.jpg"
    dest = "/home/udayanga/Desktop/output_test_samples/"
    im = cv2.imread(src)
    im = im.transpose((2,0,1))
    
    height,width = im.shape[1], im.shape[2]
    
    prefix = src.rsplit(".",1)[0]
    img_dest_full_name = dest  + "/" + src.rsplit("/",1)[1]
    ann_file = prefix + ".txt"

    ann = np.loadtxt(ann_file)
    ann = ann.reshape(-1,5)

    yolo2bbox(ann,width,height)

    #im,_ = hflip(im,ann[:,1:],height)

    im = torch.from_numpy(im)

    im,ann[:,1:] = randomcrop(im,ann[:,1:],width,height)
    
    if isinstance(im, torch.Tensor):
        im = im.numpy()

    cv2.imwrite(img_dest_full_name,im.transpose((1,2,0)))

    draw_labels_on_image(img_dest_full_name,ann,format = "bbox")"""


"""modified_images_to_matchtest_path = "."
    file_id = 0
    #frame = make_dvs_frame(events, height=260, width=346, color=False, clip=3,forDisplay = True)
    path = "/dtu/eumcaerotrain/data/latest_dataset/dset_1/npz_files_event_based/crack/crack_20/crack_20_7572871.npz"
    event_cube = get_event_cube(path)
    frame = get_single_channel_hist_from_event_cube(event_cube)
    img_array = frame
    isImgFrame = False
    save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id,img_array,isImgFrame,isvoxelgrid = False,target_size = (346,260))"""

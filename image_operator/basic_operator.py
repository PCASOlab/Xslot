import cv2
import numpy as np
import os
from random import random, choice
# import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
# import denseCRF
import os
import numpy as np
# from skimage.segmentation import (morphological_geodesic_active_contour,
#                                   inverse_gaussian_gradient)
# from utils import *
import pydensecrf.densecrf as dcrf
from skimage.transform import resize

from pydensecrf.utils import unary_from_softmax

def half_crop(video ):
    resize_to_original =  choice([True, True])
    _, D, h, w = video.shape
    mean_value = 124
    masked_video = np.full((3, D, h, w), mean_value, dtype=video.dtype)

    # Randomly select one of the four quadrants
    # quadrant = choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
    quadrant = choice(['top_left', 'top_right' ])

     
    patch = video[:, :, :h//2, :]
    masked_video[:, :, :h//2, :] = patch
  
    # Optionally resize the cropped patch back to the original dimensions
    if resize_to_original:
        patch_resized = resize(patch, (3, D, w, h), preserve_range=True).astype(video.dtype)
        return patch_resized
    else:
        return masked_video
def random_mask_or_crop(video ):
    resize_to_original =  choice([True, False])
    _, D, w, h = video.shape
    mean_value = 124
    masked_video = np.full((3, D, w, h), mean_value, dtype=video.dtype)

    # Randomly select one of the four quadrants
    # quadrant = choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
    

    
    patch = video[:, :, w//4:w//4*3, h//4:h//4*3]
    masked_video[:, :, w//4:w//4*3, h//4:h//4*3] = patch
    

    # Optionally resize the cropped patch back to the original dimensions
    if resize_to_original:
        patch_resized = resize(patch, (3, D, w, h), preserve_range=True).astype(video.dtype)
        return patch_resized
    else:
        return masked_video
def random_verse_the_video(video,max=255):
    flag = choice([True, False])
    if flag == True:
        output_v = max - video 
    else:
        output_v = video
    return output_v


def random_augment(video):
   
    ch, D, H, W= video.shape
    alpha = random() * 0.8 + 0.2  # random between 0.2 and 0.5
    angle = choice([90,180,270 ])  # random choose from 0, 90, -90

    # Reshape the stack of images to (180, 256, 256)
    images = video.reshape(ch * D, H, W)

    # Function to apply data augmentation
    def apply_data_augmentation(image):
        # Convert to PIL Image
        pil_image = Image.fromarray(np.uint8(image ))

        # Data augmentation: Random Saturation and Random Rotation
        data_augmentation = transforms.Compose([
            transforms.ColorJitter(saturation=alpha),
             
        ])
        augmented_image= transforms.functional.rotate (pil_image, angle)
        # Apply data augmentation
        augmented_image = data_augmentation(augmented_image)
         
        # Convert back to NumPy array
        augmented_image = np.array(augmented_image) 

        return augmented_image

    # Apply data augmentation to each image in the stack
    augmented_images = np.stack([apply_data_augmentation(image) for image in images])

    # Reshape augmented_images back to (3, 60, 256, 256)
    augmented_images = augmented_images.reshape(ch, D, H, W)

    return augmented_images,angle

def rotate_buff(images,angle):
    D, H, W= images.shape
  

    # Function to apply data augmentation
    def apply_data_augmentation(image):
        # Convert to PIL Image
        pil_image = Image.fromarray(np.uint8(image ))

        # Data augmentation: Random Saturation and Random Rotation
         
        augmented_image= transforms.functional.rotate (pil_image, angle)
        # Apply data augmentation
      
         
        # Convert back to NumPy array
        augmented_image = np.array(augmented_image) 

        return augmented_image

    # Apply data augmentation to each image in the stack
    augmented_images = np.stack([apply_data_augmentation(image) for image in images])

    # Reshape augmented_images back to (3, 60, 256, 256)
     

    return augmented_images 


def hide_patch(video, patch_num=32, hide_prob=0.5, mean=124,image_level= True):
    # assume patch_num is int**2
    if patch_num == 1: return video
    flag = choice([True, False])
    
    if flag == True:
       return video
    ch, D, H, W = video.shape
    pn = int(patch_num ** (1/2))
    patch_size = int(W // pn)
    patch_offsets = [(x * patch_size, y * patch_size) for x in range(pn) for y in range(pn)]

    # if np.random.uniform() < hide_prob:
    #     for d in range(D):
    #         for (px, py) in patch_offsets:
    #             video[:, d, px:px + patch_size, py:py + patch_size] = mean
    
    if image_level == False:  
        for (px, py) in patch_offsets:
            if np.random.uniform() < hide_prob:
                    video[:, :, px:px + patch_size, py:py + patch_size] = mean
    else:
      for i in range(D):
        for (px, py) in patch_offsets:
            if np.random.uniform() < hide_prob:
                    video[:, i, px:px + patch_size, py:py + patch_size] = mean
    return video


def hide_full_image(video,  hide_prob=0.5, mean=124 ):
    # assume patch_num is int**2
   
    flag = choice([True, False])
    
    if flag == True:
       return video
    ch, D, H, W = video.shape
    
    for i in range(D):
        if np.random.uniform() < hide_prob:
                video[:, i, :, :] = mean
    return video

def motion_map(video):
    ch, D, H, W = video.shape
    shifted = np.roll(video, 1, axis=1)
    motion = np.abs(video - shifted)
    motion = (motion - np.min(motion)) / np.max(motion) * 255
    return motion


def compute_optical_flow(video):
    ch, D, H, W = video.shape
    motion_map = np.zeros((D, H, W), dtype=np.float32)

    prev_frame = video[0, 0, :, :]  # Use the first frame as the initial frame

    for i in range(1, D):
        current_frame = video[0, i, :, :]
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, current_frame, flow=None, pyr_scale=0.5, levels=5, winsize=7, iterations=10, poly_n=5, poly_sigma=1.1, flags=0
        )

        # Calculate magnitude of the flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Normalize and scale the magnitude to the range [0, 255]
        magnitude = (magnitude - np.min(magnitude)) / np.max(magnitude) * 255

        # Store the computed motion map
        motion_map[i, :, :] = magnitude

        # Update the previous frame
        prev_frame = current_frame
    motion_map[0, :, :] = motion_map[1, :, :] 
    return motion_map

def DCRF(img, first_seg):
    # img = np.asarray(img)
    # img = (img*255).astype(np.uint8)

    # first_seg = first_seg.astype(np.float32)
    # prob = np.repeat(first_seg[..., np.newaxis], 2, axis=2)
    # # prob = prob[:, :, :2]
    # prob[:, :, 0] = 1.0 - prob[:, :, 0]
    # w1    = 10.0  # weight of bilateral term
    # alpha = 10    # spatial std
    # beta  = 13    # rgb  std
    # w2    = 3.0   # weight of spatial term
    # gamma = 3     # spatial std
    # it    = 50   # iteration
    # param = (w1, alpha, beta, w2, gamma, it)
    # final_seg = denseCRF.densecrf(img, prob, param)


    img = np.asarray(img)
    img = img.astype(np.uint8)

    first_seg = first_seg.astype(np.float32)
    prob = np.repeat(first_seg[np.newaxis, ...], 2, axis=0)
    # prob = prob[:, :, :2]
    prob[0, :, :] = 1.0 - prob[0, :, :]
    scale_factor = 1.0
    h, w = img.shape[:2]
    n_labels = 2
    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    img = np.ascontiguousarray(img.astype('uint8'))
    d.addPairwiseBilateral(sxy=10/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(10)
    final_seg = np.array(Q).reshape((n_labels, h, w))
    # print(final_seg.shape)
    return final_seg
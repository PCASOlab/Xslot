from eval import *
from dataset import io
from working_dir_root import Visdom_flag
import cv2
from visdom import Visdom
if Visdom_flag:
  viz = Visdom(port=8097)
from model.model_operator import post_process_softmask
from working_dir_root import Display_visdom_figure
# from  data_pre_curation. data_ytobj_box_train import apply_mask
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
# import torch.nn.functional as F
# from scipy.spatial.distance import directed_hausdorff
def scale_heatmap(heatmap):
    """
    Scale down values larger than 0.5 and scale up values between 0.1 and 0.5 in the heatmap.
    
    Args:
    heatmap (numpy array): The input heatmap with values between 0 and 1.
    
    Returns:
    numpy array: The scaled heatmap.
    """
    scaled_heatmap = np.copy(heatmap)
    
    # Scale down values larger than 0.5
    mask_high = scaled_heatmap > 0.5
    scaled_heatmap[mask_high] = 0.5 + (scaled_heatmap[mask_high] - 0.5) * 0.5  # Example scaling factor 0.5
    
    # Scale up values between 0.1 and 0.5
    mask_mid = (scaled_heatmap > 0.1) & (scaled_heatmap <= 0.5)
    scaled_heatmap[mask_mid] = 0.1+ (scaled_heatmap[mask_mid] - 0.1) * 1.8  # Example scaling factor 2.0
    
    return scaled_heatmap
def get_bounding_box(mask, threshold=0.5):
   

  # Binarize the mask (optional for non-binary masks)
  if mask.dtype != np.bool_:
    mask = mask > threshold

  contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if len(contours) == 0:
    return None

  # Get the first contour (assuming single object)
  largest_contour = max(contours, key=cv2.contourArea)

  x, y, w, h = cv2.boundingRect(largest_contour)
#   x, y, w, h = cv2.boundingRect(contour)

  return x, y, x + w, y + h
def plot_and_save_image_with_bboxes(plot_img, plot_GT_mask, plot_pr_mask):
 

  # Convert image to BGR format if needed (assuming plot_img is RGB)
  img = plot_img.copy()
  if img.shape[-1] == 3 and img.dtype == np.uint8:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  # Get bounding boxes
  gt_bbox = get_bounding_box(plot_GT_mask)
  pr_bbox = get_bounding_box(plot_pr_mask)

  # Draw bounding boxes
  if gt_bbox is not None:
    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 1)  # Green for GT
  if pr_bbox is not None:
    cv2.rectangle(img, (pr_bbox[0], pr_bbox[1]), (pr_bbox[2], pr_bbox[3]), (0, 0, 255), 1)  # Red for predicted

  # Save the image
#   cv2.imwrite(output_filename, img)
  return img,gt_bbox,pr_bbox
def plot_and_save_image_with_heatmap(plot_img, plot_pr_mask, gt_bbox,pr_bbox):
  """
  Plots a heatmap overlay of the predicted mask and bounding boxes on the image.

  Args:
      plot_img: Input image (numpy array) of shape [3, H, W].
      plot_pr_mask: Predicted mask (numpy array) of shape [H, W].
      output_filename: Path to save the output image.
  """

  # Convert image to BGR format if needed (assuming plot_img is RGB)
  img = plot_img.copy()
  if img.shape[-1] == 3 and img.dtype == np.uint8:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#   plot_pr_mask= (plot_pr_mask>0)*plot_pr_mask
#   plot_pr_mask = plot_pr_mask -np.min(plot_pr_mask)
#   plot_pr_mask = plot_pr_mask /(np.max(plot_pr_mask)+0.0000001)*254 
  plot_pr_mask = plot_pr_mask *254
    # stack
    # stack = (stack>20)*stack
    # stack = (stack>0.5)*128
  plot_pr_mask = np.clip(plot_pr_mask,0,254)
  # Apply heatmap
  heatmap = cv2.applyColorMap(plot_pr_mask.astype((np.uint8)), cv2.COLORMAP_JET)  # Scale mask to 0-255
  heatmap = cv2.addWeighted(heatmap, 0.5, img.astype((np.uint8)), 0.5, 0)  # Blend heatmap with image

  # Get bounding box (optional)
#   pr_bbox = get_bounding_box(plot_pr_mask)

  # Draw bounding box (optional)
  if gt_bbox is not None:
    cv2.rectangle(heatmap, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 1)  # Green for GT
  if pr_bbox is not None:
    cv2.rectangle(heatmap, (pr_bbox[0], pr_bbox[1]), (pr_bbox[2], pr_bbox[3]), (0, 0, 255), 1)  # Red for predicted

  # Save the image
  return heatmap
def cal_dice_np(true, predict):
    # Dice coefficient
    intersection = np.sum(true * predict)
    union = np.sum(true) + np.sum(predict)
    s = 0.000000001
    dice = (2. * intersection + s) / (union + s)
    return dice
def select_valid_masks(M_T, M_P, V_P):
  """
  Selects valid masks from ground truth, corresponding predicted masks, 
  and channel with maximum value.

  Args:
      M_T: Ground truth masks with shape [Len_array, H, W]. (Can contain None values)
      M_P: Predicted multi-channel masks with shape [channel, Len_array, H, W].
      V_P: Predicted array indicating valid channels with shape [channel].

  Returns:
      valid_M_T: Valid ground truth masks with shape [num_valid, H, W].
      valid_M_P: Corresponding predicted masks with shape [num_valid, H, W].
      max_channel: Channel index with maximum value for each valid mask with shape [num_valid].

  """
  # Find non-None indices in ground truth
  valid_indices = [i for i, mask in enumerate(M_T) if mask is not None]

  # Select valid masks from ground truth
  valid_M_T = M_T[valid_indices]

  # Select corresponding predicted masks
  valid_M_P = M_P[:, valid_indices]  # Select all channels for valid indices

  # Find channel with maximum value for each valid mask
  max_values = np.argmax(valid_M_P, axis=0)  # Find max index along channel axis

  # Select valid channels based on V_P (optional)
  # If V_P indicates specific valid channels, uncomment this:
  # valid_channels = V_P[max_values]

  return valid_M_T, valid_M_P, max_values
 
# Example usage
# cal_all_metrics(...)
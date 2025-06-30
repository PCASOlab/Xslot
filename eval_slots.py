from eval import *
from dataset import io
from working_dir_root import Visdom_flag,Save_flag
from sklearn.metrics import adjusted_rand_score
import cv2
import os
from visdom import Visdom
from display import stack_to_color_mask

if Visdom_flag:
  viz = Visdom(port=8097)
from model.model_operator import post_process_softmask
from working_dir_root import Display_visdom_figure
# from  data_pre_curation. data_ytobj_box_train import apply_mask
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label
 
from scipy.spatial.distance import directed_hausdorff
# Original base colors (do not change order)
base_colors = [
    (128, 0, 128),    # Purple
    (0, 128, 0),      # Green
    (0, 255, 255),    # Cyan
    (255, 0, 0),      # Red
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 165, 0)     # Orange
]

# Extend with additional distinct colors
extended_colors = base_colors + [
    (75, 0, 130),     # Indigo
    (255, 192, 203),  # Pink
    (0, 255, 127),    # Spring Green
    (173, 255, 47),   # Green Yellow
    (139, 69, 19),    # Saddle Brown
    (70, 130, 180)    # Steel Blue
]
def binary_to_multi_channel(binary_mask):
    """
    Convert a binary mask into a multi-channel mask, where each channel represents a distinct object.

    Args:
        binary_mask: A 2D numpy array of shape (H, W) where pixels are either 0 or 1.

    Returns:
        multi_channel_mask: A 3D numpy array of shape (N, H, W) where N is the number of distinct objects.
    """
    # Define the structure for 4-connectivity
    structure = np.array([[0, 1, 0], 
                          [1, 1, 1], 
                          [0, 1, 0]], dtype=np.int8)

    # Apply connected component labeling with 4-connectivity
    labeled_mask, num_features = label(binary_mask, structure=structure)  # Label connected components

    # Create multi-channel mask
    multi_channel_mask = np.zeros((num_features, *binary_mask.shape), dtype=np.float32)

    for i in range(1, num_features + 1):  # Start from 1 to ignore background (0)
        multi_channel_mask[i - 1] = (labeled_mask == i).astype(np.float32)  # Create a binary mask for the current object

    return multi_channel_mask

def convert_label_frame_to_instance_masks(label_frame, min_gap_size=5):
    """
    Convert a binary label frame to instance masks and fill small gaps.

    Args:
        label_frame: A tensor of shape (N, H, W) containing binary masks.
        min_gap_size: The size of small gaps to ignore (in pixels).

    Returns:
        instance_masks: A tensor of shape (num_instances, H, W) representing instance masks.
    """
    N, H, W = label_frame.size()  # Get the dimensions of the label frame
    instance_masks = []

    # Structuring element for morphological closing (filling gaps)
    kernel = np.ones((min_gap_size, min_gap_size), np.uint8)

    # Iterate through each channel (N dimension)
    for channel_idx in range(N):
        binary_mask = label_frame[channel_idx].cpu().numpy()  # Get the binary mask for the channel
        
        # Perform morphological closing to fill small gaps
        processed_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Create multi-channel mask for the processed binary mask
        multi_channel_mask = binary_to_multi_channel(processed_mask)

        # Append the multi-channel masks for this channel to the instance masks
        instance_masks.extend(multi_channel_mask)  # Extend to include all new instance masks

    return torch.tensor(instance_masks)  # Convert the list back to a tensor
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
# import torch.nn.functional as F
# from scipy.spatial.distance import directed_hausdorff
def remove_empty_channels(mask_stack, threshold=10):
    """
    Remove channels that have fewer than the threshold of non-zero pixels from the ground truth mask stack.
    Args:
        mask_stack: Ground truth mask stack, shape [n, L, H, W].
        threshold: Minimum number of non-zero pixels for a channel to be considered non-empty (default is 10).
    Returns:
        filtered_mask_stack: Mask stack with channels having at least 'threshold' non-zero pixels, shape [n_filtered, L, H, W].
    """
    # Count the number of non-zero pixels in each channel
    non_zero_counts = (mask_stack > 0).sum(dim=[1, 2, 3])  # Count non-zero pixels in each channel

    # Get indices of channels with at least 'threshold' non-zero pixels
    non_empty_channels = torch.nonzero(non_zero_counts >= threshold, as_tuple=False).squeeze(1)

    # Select only non-empty channels
    filtered_mask_stack = mask_stack[non_empty_channels, :, :, :]

    return filtered_mask_stack
def remove_empty_channels_frame(mask_stack, threshold=20):
    """
    Remove channels that have fewer than the threshold of non-zero pixels from the ground truth mask stack.
    Args:
        mask_stack: Ground truth mask stack, shape [n, L, H, W].
        threshold: Minimum number of non-zero pixels for a channel to be considered non-empty (default is 10).
    Returns:
        filtered_mask_stack: Mask stack with channels having at least 'threshold' non-zero pixels, shape [n_filtered, L, H, W].
    """
    # Count the number of non-zero pixels in each channel
    if not mask_stack.any():
        print ("no instance ")
        return mask_stack
    non_zero_counts = (mask_stack > 0).sum(dim=[1, 2])  # Count non-zero pixels in each channel

    # Get indices of channels with at least 'threshold' non-zero pixels
    non_empty_channels = torch.nonzero(non_zero_counts >= threshold, as_tuple=False).squeeze(1)

    # Select only non-empty channels
    filtered_mask_stack = mask_stack[non_empty_channels, :, :]

    return filtered_mask_stack



def hungarian_dice(label_mask, predic_mask_3D):
    """
    Calculate the minimal Dice coefficient using the Hungarian algorithm between ground truth and predicted masks.
    
    Args:
        label_mask: Ground truth masks (N, L, H, W)
        predic_mask_3D: Predicted masks (M, L, H, W)

    Returns:
        avg_dice: Average minimal Dice coefficient for the best matching masks
    """
    N, L, H, W = label_mask.size()
    M, _, _, _ = predic_mask_3D.size()
    if torch.isnan(label_mask).any():
        return np.nan,np.nan
    # Initialize Dice matrix
    dice_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            dice_matrix[i, j] = cal_dice(label_mask[i], predic_mask_3D[j]).item()

    # Apply Hungarian algorithm to maximize the Dice coefficient matching
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    # Compute the average minimal Dice coefficient
    avg_dice = dice_matrix[row_ind, col_ind].mean()

    return avg_dice, dice_matrix[row_ind, col_ind]
def hungarian_iou_per_frame(label_frame, predic_frame):
    """
    Calculate the max matching IoU using the Hungarian algorithm for a single frame.
    
    Args:
        label_frame: Ground truth mask for a single frame, shape (N, H, W).
        predic_frame: Predicted mask for a single frame, shape (M, H, W).

    Returns:
        avg_iou: Average max IoU for the best matching masks for this frame.
    """
    N, H, W = label_frame.size()
    M, _, _ = predic_frame.size()
    if torch.isnan(label_frame).any():
            return np.nan
    # Initialize IoU matrix for this frame
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            # Calculate IoU between the ith ground truth mask and jth predicted mask for the frame
            iou_matrix[i, j] = cal_J(label_frame[i], predic_frame[j]).item()

    # Apply Hungarian algorithm to maximize the IoU matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Compute the average IoU for this frame
    avg_iou = iou_matrix[row_ind, col_ind].mean()

    return avg_iou


def hungarian_dice_per_frame(label_frame, predic_frame):
    """
    Calculate the max matching Dice coefficient using the Hungarian algorithm for a single frame.
    
    Args:
        label_frame: Ground truth mask for a single frame, shape (N, H, W).
        predic_frame: Predicted mask for a single frame, shape (M, H, W).

    Returns:
        avg_dice: Average max Dice coefficient for the best matching masks for this frame.
    """
    N, H, W = label_frame.size()
    M, _, _ = predic_frame.size()
    if torch.isnan(label_frame).any():
            return np.nan
    # Initialize Dice matrix for this frame
    dice_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            dice_matrix[i, j] = cal_dice(label_frame[i], predic_frame[j]).item()

    # Apply Hungarian algorithm to maximize the Dice matching
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    # Compute the average Dice coefficient for this frame
    avg_dice = dice_matrix[row_ind, col_ind].mean()

    return avg_dice
from torchvision.ops import box_iou
def mask_to_bbox(mask):
    """
    Convert a binary mask to a bounding box.
    Args:
        mask: Binary mask, shape (H, W).
    Returns:
        bbox: Bounding box in (x_min, y_min, x_max, y_max) format.
    """
    mask_np = mask.cpu().numpy()
    indices = np.argwhere(mask_np > 0)

    if len(indices) == 0:  # No foreground pixels
        return (0, 0, 0, 0)

    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return (x_min, y_min, x_max, y_max)

def hungarian_iou_per_frame_instance(label_frame, predic_frame):
    """
    Calculate the max matching Dice coefficient, Hausdorff distance, and matched masks using the Hungarian algorithm,
    along with the percentage of matched masks with box IoU > 0.5.

    Args:
        label_frame: Ground truth mask for a single frame, shape (N, H, W).
        predic_frame: Predicted mask for a single frame, shape (M, H, W).

    Returns:
        avg_dice: Average max Dice coefficient for the best matching masks for this frame.
        avg_hd: Average Hausdorff distance for the best matching masks for this frame.
        matched_gt_masks: Matched ground truth masks, shape (N, H, W) after matching.
        matched_pred_masks: Matched predicted masks, shape (M, H, W) after matching.
        matched_box_iou_percentage: Percentage of matched masks with box IoU > 0.5.
    """
    if torch.isnan(label_frame).any():
        return np.nan, np.nan, None, None, np.nan

    # Convert ground truth to instance masks
    instance_masks = convert_label_frame_to_instance_masks(label_frame)
    instance_masks = remove_empty_channels_frame(instance_masks)

    if not instance_masks.any():
        print("NO GT")
        return np.nan, np.nan, None, None, np.nan  # No ground truth, return NaN for metrics

    N, H, W = instance_masks.size()
    instance_masks = instance_masks.to(predic_frame.device)
    M, _, _ = predic_frame.size()

    # Initialize Dice and Hausdorff distance matrices for this frame
    dice_matrix = np.zeros((N, M))
    hausdorff_matrix = np.zeros((N, M))

    # Compute mask-level Dice and Hausdorff distance
    for i in range(N):
        for j in range(M):
            # Dice coefficient
            dice_matrix[i, j] = cal_J(instance_masks[i], predic_frame[j]).item()

            # Hausdorff distance
            gt_indices = np.argwhere(instance_masks[i].cpu().numpy() != 0)  # Non-background pixels in GT mask
            pred_indices = np.argwhere(predic_frame[j].cpu().numpy() != 0)  # Non-background pixels in predicted mask

            if len(gt_indices) > 0 and len(pred_indices) > 0:
                hausdorff_matrix[i, j] = max(directed_hausdorff(gt_indices, pred_indices)[0],
                                             directed_hausdorff(pred_indices, gt_indices)[0])
            else:
                hausdorff_matrix[i, j] = np.nan

    # Apply Hungarian algorithm to maximize Dice matching
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    # Compute matched Dice and Hausdorff averages
    avg_dice = dice_matrix[row_ind, col_ind].mean()
    avg_hd = np.nanmean(hausdorff_matrix[row_ind, col_ind])

    # Compute matched ground truth and predicted masks
    matched_gt_masks = torch.stack([instance_masks[i] for i in row_ind], dim=0)
    matched_pred_masks = torch.stack([predic_frame[j] for j in col_ind], dim=0)

    # Calculate bounding boxes for matched masks
    gt_boxes = [mask_to_bbox(m) for m in matched_gt_masks]
    pred_boxes = [mask_to_bbox(m) for m in matched_pred_masks]

    # Convert bounding boxes to tensors
    gt_boxes_tensor = torch.tensor(gt_boxes, device=predic_frame.device)
    pred_boxes_tensor = torch.tensor(pred_boxes, device=predic_frame.device)

    # Calculate box IoU
    box_ious = box_iou(gt_boxes_tensor, pred_boxes_tensor)

    # Count matches with box IoU > 0.5
    matched_box_iou_count = (box_ious.diag() > 0.5).sum().item()

    # Calculate percentage of matched masks with box IoU > 0.5
    matched_box_iou_percentage = matched_box_iou_count / N if N > 0 else 0

    return avg_dice, avg_hd, matched_gt_masks, matched_pred_masks, matched_box_iou_percentage
def calculate_iou(pred, gt):
    """Calculate IoU between two binary masks."""
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 0
    else:
        return intersection / union

def hungarian_matching(pred_masks, gt_masks):
    """
    Perform Hungarian matching based on IoU between predicted and ground truth masks.
    
    Args:
        pred_masks: Predicted binary masks, shape (P, H * W).
        gt_masks: Ground truth binary masks, shape (G, H * W).
        
    Returns:
        matched_pred: Matched predicted masks, reordered based on ground truth.
        matched_gt: Ground truth masks, possibly duplicated.
    """
    P, H_W = pred_masks.shape
    G, _ = gt_masks.shape

    # Compute IoU matrix between all ground truth and predicted masks
    iou_matrix = np.zeros((G, P))
    for g in range(G):
        for p in range(P):
            iou_matrix[g, p] = calculate_iou(gt_masks[g], pred_masks[p])
    
    # Perform Hungarian matching to maximize IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximizing IoU
    
    # Reorder predicted masks according to Hungarian matching
    matched_pred = pred_masks[col_ind]
    matched_gt = gt_masks[row_ind]
    
    return matched_pred, matched_gt

def get_ari_multichannel(prediction_masks, gt_masks, bg_class=0):
    """
    Calculate ARI for multi-channel binary masks after Hungarian matching.
    
    Args:
        prediction_masks: Predicted masks, shape (P, H, W).
        gt_masks: Ground truth masks, shape (G, H, W).
        bg_class: Background class, usually 0.
        
    Returns:
        ari: Adjusted Rand Index for the matched masks.
    """
    # Flatten masks along the spatial dimensions
    prediction_masks_flat = prediction_masks.flatten(start_dim=1).cpu().numpy().astype(int)
    gt_masks_flat = gt_masks.flatten(start_dim=1).cpu().numpy().astype(int)
    
    # Perform Hungarian matching based on IoU
    matched_pred, matched_gt = hungarian_matching(prediction_masks_flat, gt_masks_flat)
    
    # Compute ARI for each frame
    rand_scores = []
    rand_scores=adjusted_rand_score(matched_gt.flatten(), matched_pred.flatten())
    # for pred, gt in zip(matched_pred, matched_gt):
    #     if np.all(gt == bg_class):  # Skip if the ground truth is all background
    #         continue
    #     rand_scores.append(adjusted_rand_score(gt, pred))
    
    # Average ARI score across frames
    # if len(rand_scores) == 0:
    if  rand_scores is None:

        ari = np.nan
    else:
        # ari = sum(rand_scores) / len(rand_scores)
        ari = rand_scores
    
    return ari

def get_ari_multichannel2(prediction_masks, gt_masks, bg_class=0):
    """
    Calculate ARI for multi-channel binary masks after Hungarian matching.
    
    Args:
        prediction_masks: Predicted masks, shape (P, H, W).
        gt_masks: Ground truth masks, shape (G, H, W).
        bg_class: Background class, usually 0.
        
    Returns:
        ari: Adjusted Rand Index for the matched masks.
    """
    # Flatten masks along the spatial dimensions
    prediction_masks_flat = prediction_masks.flatten(start_dim=1).cpu().numpy().astype(int)
    gt_masks_flat = gt_masks.flatten(start_dim=1).cpu().numpy().astype(int)
    
    # Perform Hungarian matching based on IoU
    matched_pred, matched_gt = hungarian_matching(prediction_masks_flat, gt_masks_flat)
    
    # Compute ARI for each frame
    rand_scores = []
    for pred, gt in zip(matched_pred, matched_gt):
        if np.all(gt == bg_class):  # Skip if the ground truth is all background
            continue
        rand_scores.append(adjusted_rand_score(gt, pred))
    
    # Average ARI score across frames
    if len(rand_scores) == 0:
        ari = np.nan
    else:
        ari = sum(rand_scores) / len(rand_scores)
    
    return ari

def process_metrics_from_excel(excel_path, output_root):
    # Load the Excel file
    metrics_df = pd.read_excel(excel_path)

    # Extract the required metrics
    metrics_df = metrics_df[['read_id', 'IoU', 'Frame_level_iou_instance', 'Frame-level average HD instance', 'Frame-level average ARI','Frame-level average corloc']]

    # Initialize lists for groups
    group_averages = []
    current_group = []

    # Iterate through each row to detect new groups based on 'read_id'
    for _, row in metrics_df.iterrows():
        read_id = row['read_id']
        
        # If read_id is 0 and we already have a group collected, finalize the current group and start a new one
        if int(read_id) == 0 and current_group:
            # Calculate the mean for the current group
            group_mean = pd.DataFrame(current_group).mean().values
            group_averages.append(group_mean)
            current_group = []  # Reset for the next group
        
        # Add the current row's metrics to the current group
        current_group.append(row[['IoU', 'Frame_level_iou_instance', 'Frame-level average HD instance', 'Frame-level average ARI','Frame-level average corloc']].values)

    # Append the final group if it has data
    if current_group:
        group_mean = pd.DataFrame(current_group).mean().values
        group_averages.append(group_mean)

    # Debug: Print group averages to see their values
    print("Group Averages:\n", group_averages)

    # Convert the list of group averages into a DataFrame
    group_averages_df = pd.DataFrame(group_averages, columns=['IoU', 'Frame_level_iou_instance', 'Frame-level average HD instance', 'Frame_level average ARI','Frame-level average corloc'])

    # Debug: Check if any columns in the DataFrame are NaN
    print("Group Averages DataFrame:\n", group_averages_df)

    # Calculate the overall average using all data points
    overall_average = metrics_df[['IoU', 'Frame_level_iou_instance', 'Frame-level average HD instance', 'Frame-level average ARI','Frame-level average corloc']].mean()

    # Calculate the standard deviation between the group averages
    std_between_groups = group_averages_df.std()

    # Prepare a summary DataFrame for saving
    summary_df = pd.DataFrame({
        'Metric': ['IoU', 'Frame_level_iou_instance', 'Frame_level average HD instance', 'Frame_level average ARI','Frame-level average corloc'],
        'Overall Average': overall_average.round(3).values,  # Round to 3 decimal places
        'Standard Deviation Between Groups': std_between_groups.round(3).values  # Round to 3 decimal places
    })
    # Create a new column with mean and std together using "±"
    summary_df['Mean ± Std'] = summary_df.apply(
        lambda row: f"{row['Overall Average']} ± {row['Standard Deviation Between Groups']}", axis=1
    )
    print("Mean ± Std for each metric:")
    for value in summary_df['Mean ± Std']:
        print(value)
    # Create output directory if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Define the output file path
    output_file = os.path.join(output_root, 'processed_metrics_summary.xlsx')

    # Save to Excel with 3 decimal places
    summary_df.to_excel(output_file, index=False, float_format='%.3f', sheet_name='Summary')

    print(f"Processed metrics summary has been saved to {output_file}")
    return output_file
# frame = frame.transpose(1, 2, 0)  # Convert from (3, H, W) -> (H, W, 3)
# 
def overlap_multichannel_gt_pred_separate(frame, mask_gt, mask_pred, gt_channel_colors, 
                                        pred_color_dict, alpha=0.5):
    # GT overlay using gt_channel_colors
    frame = frame.transpose(1, 2, 0) 
    mask_gt = mask_gt.cpu().numpy()
    mask_pred = mask_pred.cpu().numpy()
    combined_color_mask_gt = np.zeros_like(frame)
    for c in range(mask_gt.shape[0]):
        color = gt_channel_colors[c]
        combined_color_mask_gt[mask_gt[c] == 1] = color
    
    # Pred overlay using pred_color_dict (already has GT colors for matches)
     # Convert masks to CPU numpy arrays
    # mask_gt = mask_gt.cpu().numpy()
    # mask_pred = mask_pred.cpu().numpy()

    # Initialize blended frames
    blended_gt_frame = frame.copy()
    blended_pred_frame = frame.copy()

    combined_color_mask_pred = np.zeros_like(frame)
    for c in range(mask_pred.shape[0]):
        color = pred_color_dict[c]
        combined_color_mask_pred[mask_pred[c] == 1] = color
    
    # Blend both masks
    blended_gt = cv2.addWeighted(frame, 1-alpha, combined_color_mask_gt, alpha, 0)
    blended_pred = cv2.addWeighted(frame, 1-alpha, combined_color_mask_pred, alpha, 0)
    
    return blended_gt, blended_pred
def select_non_nan_masks(filtered_label_mask, predic_mask_3D,input_video):
    # Find indices along D dimension where there are no NaNs in filtered_label_mask
    non_nan_indices = [i for i in range(filtered_label_mask.shape[1]) 
                       if not torch.isnan(filtered_label_mask[:, i, :, :]).any()]

    # Select the non-NaN parts from both masks
    filtered_label_mask_non_nan = filtered_label_mask[:, non_nan_indices, :, :]
    predic_mask_3D_non_nan = predic_mask_3D[:, non_nan_indices, :, :]
    video_nonan =  input_video[:, non_nan_indices, :, :]

    return filtered_label_mask_non_nan, predic_mask_3D_non_nan,video_nonan,non_nan_indices
def hungarian_iou(label_mask, predic_mask_3D):
    """
    Calculate the minimal IoU using the Hungarian algorithm between ground truth and predicted masks.
    
    Args:
        label_mask: Ground truth masks (N, L, H, W)
        predic_mask_3D: Predicted masks (M, L, H, W)

    Returns:
        avg_iou: Average minimal IoU for the best matching masks
    """
    N, L, H, W = label_mask.size()  # N = number of ground truth channels
    M, _, _, _ = predic_mask_3D.size()  # M = number of predicted channels
    if torch.isnan(label_mask).any():
        return np.nan,np.nan
    # Initialize IoU matrix
    iou_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            # Calculate IoU between the ith ground truth mask and jth predicted mask
            iou_matrix[i, j] = cal_J(label_mask[i], predic_mask_3D[j]).item()

    # Apply Hungarian algorithm to maximize the IoU matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # We negate the matrix because we want to maximize

    # Compute the average minimal IoU
    avg_iou = iou_matrix[row_ind, col_ind].mean()
    matched_gt_video = label_mask[row_ind, :, :, :]  # Using row_ind from Hungarian matching
    matched_pred_video = predic_mask_3D[col_ind, :, :, :]  # Using col_ind from Hungarian matching

    return avg_iou, iou_matrix[row_ind, col_ind],matched_gt_video,matched_pred_video,row_ind, col_ind
def custom_stack_to_color_mask(mask_stack, H, W, pred_color_dict, matched_channels, 
                              alpha_matched=0.8, alpha_unmatched=0.3):
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for c in range(mask_stack.shape[0]):
        binary_mask = mask_stack[c].astype(np.uint8)
        color = pred_color_dict[c]
        is_matched = c in matched_channels
        
        alpha = alpha_matched if is_matched else alpha_unmatched
        
        color_layer = np.zeros_like(color_mask)
        color_layer[binary_mask > 0] = color
        color_mask = cv2.addWeighted(color_mask, 1, color_layer, alpha, 0)
        
        # Add border for matched
        if is_matched:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_mask, contours, -1, color.tolist(), 2)
    
    return color_mask

def cal_all_metrics_slots(read_id, Output_root, label_mask, predic_mask_3D,input_video_OG):
    mask_threshold = 0.5
    if not label_mask.any():
        print ("NO GT")
        return
    device = label_mask.device
    predic_mask_3D = predic_mask_3D.to(device)
    _, D, H, W = label_mask.size()
    ch_ini, _,_,_ = predic_mask_3D.size()
    predic_mask_3D = F.interpolate(predic_mask_3D, size=(H, W), mode='bilinear', align_corners=False)
    display_3d_OG = predic_mask_3D>0.35
    predic_mask_3D = (predic_mask_3D > 0) * predic_mask_3D
    predic_mask_3D = predic_mask_3D - torch.min(predic_mask_3D)
    predic_mask_3D = predic_mask_3D / (torch.max(predic_mask_3D) + 0.0000001) * 1
    predic_mask_3D = predic_mask_3D > mask_threshold
    predic_mask_3D = torch.clamp(predic_mask_3D, 0, 1)
    filtered_label_mask = remove_empty_channels(label_mask) 
    filtered_label_mask, predic_mask_3D,input_video,non_nan_indices=select_non_nan_masks(filtered_label_mask, predic_mask_3D,input_video_OG)
    display_3d = display_3d_OG[:,non_nan_indices,:,:]
    ch, D, H, W = filtered_label_mask.size()
    _,D_OG,_,_ = display_3d_OG.size()
    # Calculate minimal IoU and Dice using Hungarian Matching over entire video
    avg_iou, matched_ious,matched_gt_video,matched_pred_video,row_ind, col_ind= hungarian_iou(filtered_label_mask, predic_mask_3D)
    # Get the matched GT and predicted masks for the entire video
    # matched_gt_video = filtered_label_mask[row_ind, :, :, :]  # Using row_ind from Hungarian matching
    # matched_pred_video = predic_mask_3D[col_ind, :, :, :]  # Using col_ind from Hungarian matching

    
     # Process each frame
    
    avg_dice, matched_dices = hungarian_dice(filtered_label_mask, predic_mask_3D)
    # Frame-level IoU and Dice calculation
    frame_level_ious = []
    frame_level_dices = []
    frame_level_iou_instance=[]
    frame_level_HD_instance=[]
    frame_level_corloc_instance=[]
    
    frame_level_ari=[]
    video_stack_gt = []
    video_stack_pred = []


    for frame_idx in range(D):
        frame = input_video[:,frame_idx,:,:]  # (3,256,256)

        label_frame = filtered_label_mask[:, frame_idx, :, :]
        predic_frame = predic_mask_3D[:, frame_idx, :, :]

        avg_iou_frame = hungarian_iou_per_frame(label_frame, predic_frame)
        avg_dice_frame = hungarian_dice_per_frame(label_frame, predic_frame)

        avg_iou_frame_instance,avg_HD_frame_instance,match_GT,matched_pred, corloc= hungarian_iou_per_frame_instance(label_frame, predic_frame)
        avg_dice_frame_ari = get_ari_multichannel(label_frame, predic_frame)
        

        frame_level_ious.append(avg_iou_frame)
        frame_level_dices.append(avg_dice_frame)
        frame_level_iou_instance.append(avg_iou_frame_instance)
        frame_level_HD_instance.append(avg_HD_frame_instance)
        frame_level_ari.append(avg_dice_frame_ari)
        frame_level_corloc_instance.append(corloc)


        # if match_GT is not None and matched_pred is not None:
        #     # Overlap ground truth and predicted masks, ensuring consistent colors between them
        #     blended_gt_frame, blended_pred_frame,channel_colors= overlap_multichannel_gt_pred_separate(
        #         frame, match_GT, matched_pred, alpha=0.5)
            
        #     video_stack_gt.append(blended_gt_frame)
        #     video_stack_pred.append(blended_pred_frame)
    # if len(video_stack_gt) >0 and len (video_stack_pred) >0:
    #     video_stack_gt = np.hstack(video_stack_gt)  # Stack all GT overlays
    #     video_stack_pred = np.hstack(video_stack_pred)  # Stack all predicted overlays

    #     combine_stack = np.vstack([video_stack_gt, video_stack_pred])

    # Transpose to put the color channels in the correct position for displaying
        # combine_stack = combine_stack.transpose(1, 2, 0)

        # viz.image(np.transpose(combine_stack.astype((np.uint8)), (2, 0, 1)), opts=dict(title=f'{read_id} - stack_color_mask'))
    if Save_flag:
        # Initialize lists for stacking matched color overlays
        num_categories = 11
        category_colors = extended_colors[:num_categories]

        # Convert to numpy array for indexing
        gt_channel_colors = np.array(category_colors[:filtered_label_mask.size(0)])
        # gt_channel_colors = {i: np.random.randint(0, 255, 3) for i in range(filtered_label_mask.size(0))}

        # pred_color_dict = {i: np.random.randint(0, 255, 3) for i in range(num_pred_channels)}
        # Create predicted color dict: matched channels use GT color, others random
        pred_color_dict = {}
        num_pred_channels = predic_mask_3D.size(0)
        um_channel_colors =  np.array(category_colors[filtered_label_mask.size(0):len(category_colors)])
        for pred_idx in range(num_pred_channels):
            if pred_idx in col_ind:
                # Find corresponding GT index
                gt_idx = row_ind[np.where(col_ind == pred_idx)[0][0]]
                pred_color_dict[pred_idx] =  np.array(gt_channel_colors[gt_idx])
            else:
                # Select an unmatched color from extended list
                unmatched_idx = (pred_idx % len(um_channel_colors))  # Cycle through colors
                pred_color_dict[pred_idx] =  np.array(um_channel_colors[unmatched_idx])


        # Track which predicted channels were matched
        matched_channels = set(col_ind)

        # Modify visualization loop
        video_stack_all_color = []
        
        
        video_stack_matched_gt = []
        video_stack_matched_pred = []
        video_stack_origin_valid = []
        video_stack_all_color  = []

        channel_colors = None
        # Define paths for saving frames
        matched_gt_folder = os.path.join(Output_root, "image/match_color_mask", str(read_id), "matched_gt")
        all_color_folder = os.path.join(Output_root, "image/match_color_mask", str(read_id), "all_color")
        # Ensure directories exist
        os.makedirs(matched_gt_folder, exist_ok=True)
        os.makedirs(all_color_folder, exist_ok=True)
        for frame_idx in range(D):
            frame = input_video[:, frame_idx, :, :]  # (3, H, W)

            matched_gt_frame = matched_gt_video[:, frame_idx, :, :]
            matched_pred_frame = matched_pred_video[:, frame_idx, :, :]

            # Overlap matched ground truth and predicted masks with the input frame
            blended_gt_frame, blended_pred_frame = overlap_multichannel_gt_pred_separate(
                frame, matched_gt_frame, matched_pred_frame,gt_channel_colors ,pred_color_dict, alpha=0.5
            )
            pred_masks_frame = display_3d[:, frame_idx, :, :].cpu().numpy()

# Generate color mask using custom coloring
            frame_color_mask = custom_stack_to_color_mask(
                pred_masks_frame, H, W,
                pred_color_dict=pred_color_dict,
                matched_channels=matched_channels
            )
            blended = cv2.addWeighted(frame.transpose(1, 2, 0), 0.5, frame_color_mask, 0.5, 0)
            video_stack_all_color.append(blended)
            video_stack_matched_gt.append(blended_gt_frame)
            # video_stack_matched_pred.append(blended_pred_frame)
    #         if frame.shape[0] == 3:
    # frame = frame.transpose(1, 2, 0)  # Convert from (3, H, W) -> (H, W, 3)
            video_stack_origin_valid.append(frame.transpose(1, 2, 0))


        # longer unsampled
        undownsampled_color_folder = os.path.join(Output_root, "image/match_color_mask", str(read_id), "undownsampled_color")
        os.makedirs(undownsampled_color_folder, exist_ok=True)
        for frame_idx2 in range(D_OG):
            frame2 = input_video_OG[:, frame_idx2, :, :]  # (3, H, W)
 
            pred_masks_frame2 = display_3d_OG[:, frame_idx2, :, :].cpu().numpy()

# Generate color mask using custom coloring
            frame_color_mask2 = custom_stack_to_color_mask(
                pred_masks_frame2, H, W,
                pred_color_dict=pred_color_dict,
                matched_channels=matched_channels
            )
            blended2 = cv2.addWeighted(frame2.transpose(1, 2, 0), 0.5, frame_color_mask2, 0.5, 0)
            undownsampled_frame_path = os.path.join(undownsampled_color_folder, f"frame_{frame_idx2:04d}.png")
            cv2.imwrite(undownsampled_frame_path, blended2.astype(np.uint8))
        # Save each frame separately
        for frame_idx in range(D):
            gt_frame_path = os.path.join(matched_gt_folder, f"frame_{frame_idx:04d}.png")
            color_frame_path = os.path.join(all_color_folder, f"frame_{frame_idx:04d}.png")

            cv2.imwrite(gt_frame_path, video_stack_matched_gt[frame_idx].astype(np.uint8))
            cv2.imwrite(color_frame_path, video_stack_all_color[frame_idx].astype(np.uint8))
        if video_stack_matched_gt  :
            video_stack_matched_gt = np.hstack(video_stack_matched_gt)  # Stack GT overlays
            # video_stack_matched_pred = np.hstack(video_stack_matched_pred)  # Stack predicted overlays
            video_stack_origin_valid = np.hstack(video_stack_origin_valid)
            video_stack_all_color = np.hstack(video_stack_all_color)
            
            # all_color_mask = stack_to_color_mask (display_3d.cpu().numpy(),H,W,np.ones(ch_ini),np.ones(ch_ini),1)
            # alpha= 0.5
            # video_stack_all_color = cv2.addWeighted(video_stack_origin_valid.astype((np.uint8)), 1 - alpha, all_color_mask.astype((np.uint8)), alpha, 0)
            # video_stack_all_color. append(stack_color_mask)
            # video_stack_all_color = np.hstack(video_stack_all_color)

            combine_stack_matched = np.vstack([video_stack_origin_valid,video_stack_matched_gt,video_stack_all_color])
        
        
            io.save_img_to_folder(Output_root + "image/match_color_mask/" ,  read_id, combine_stack_matched.astype((np.uint8)) )
    # cv2.imshow("matched masks overlay", combine_stack.transpose)

    avg_frame_level_iou = np.nanmean(frame_level_ious)
    avg_frame_level_dice = np.nanmean(frame_level_dices)
    avg_frame_level_iou_instance = np.nanmean(frame_level_iou_instance)
    avg_frame_level_HD_instance = np.nanmean(frame_level_HD_instance)

    avg_frame_level_ari = np.nanmean(frame_level_ari)

    avg_frame_level_corloc= np.nanmean(frame_level_corloc_instance)


    print(f"Average max IoU (Hungarian): {avg_iou:.4f}")
    print(f"Average max Dice (Hungarian): {avg_dice:.4f}")
    print(f"Frame-level average max IoU (Hungarian): {avg_frame_level_iou:.4f}")
    print(f"Frame-level average max Dice (Hungarian): {avg_frame_level_dice:.4f}")
    print(f"Frame-level average max IOU instance (Hungarian): {avg_frame_level_iou_instance:.4f}")
    print(f"Frame-level average HD instance (Hungarian): {avg_frame_level_HD_instance:.4f}")
    print(f"Frame-level average ARI: {avg_frame_level_ari:.4f}")
    print(f"Frame-level average corloc: {avg_frame_level_corloc:.4f}")



    global metrics_video_data
    metrics_video_data.append({
        'read_id': read_id,
        'IoU': avg_iou,
        'Dice_Coefficient': avg_dice,
        'Frame_level_IoU': avg_frame_level_iou,
        'Frame_level_Dice': avg_frame_level_dice,
        'Frame_level_iou_instance': avg_frame_level_iou_instance,
        'Frame-level average HD instance':avg_frame_level_HD_instance,
        'Frame-level average ARI':avg_frame_level_ari,
        'Frame-level average corloc': avg_frame_level_corloc,
        # Add other metrics here if needed
    })

    metrics_video = pd.DataFrame(metrics_video_data)
    if not os.path.exists(Output_root):
        os.makedirs(Output_root)
    metrics_video.to_excel(Output_root + 'metrics_video.xlsx', index=False, float_format='%.4f')
 
if __name__ == "__main__":
    from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
    from working_dir_root import Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
    from working_dir_root import Enable_student,Batch_size,selected_data,Display_down_sample, Data_percentage,Gpu_selection,Evaluation_slots,Max_epoch,Output_root
    print("Hello, World!")
    Output_root = Output_root+ "Obj_centric_temp2MLP_BERT_mask_feat" + selected_data + str(Data_percentage) + "/"
    excel_path = Output_root + 'metrics_video.xlsx'
# Usage example
# Specify the path to your original Excel file, the value for N, and the output root directory
# excel_path = 'Output_root/metrics_video.xlsx'
# output_root = 'Output_root'
    N = 99  # Adjust N as needed
    output_file = process_metrics_from_excel(excel_path, Output_root)
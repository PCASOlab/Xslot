import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
# Create empty DataFrames to store metrics
metrics_video_data = []
metrics_frame_data = []
def cal_hausdorff(true, predict, class_indices):
    hausdorff_distances = []
    
    true_np = true.cpu().numpy()  # Convert to numpy array
    predict_np = predict.cpu().numpy()  # Convert to numpy array

    for class_idx in class_indices:
        for frame in range(true_np.shape[1]):  # Iterate over all frames
            true_frame = true_np[class_idx, frame]
            predict_frame = predict_np[class_idx, frame]
            
            # Get the coordinates of the true and predicted points
            true_points = np.argwhere(true_frame > 0)
            predict_points = np.argwhere(predict_frame > 0)
            
            if len(true_points) == 0 or len(predict_points) == 0:
                continue  # Skip frames with no points

            # Calculate the directed Hausdorff distance
            hausdorff_dist = max(directed_hausdorff(true_points, predict_points)[0],
                                 directed_hausdorff(predict_points, true_points)[0])
            hausdorff_distances.append(hausdorff_dist)

    if hausdorff_distances:
        average_hausdorff_distance = np.mean(hausdorff_distances)
    else:
        average_hausdorff_distance = 0  # If no valid frames, return infinity

    return average_hausdorff_distance
def cal_J(true, predict):
    # Intersection and Union for calculating Jaccard Index (Intersection over Union)
    AnB = true * predict  # Element-wise multiplication for intersection
    AuB = true + predict  # Element-wise addition for union
    AuB = torch.clamp(AuB, 0, 1)  # Clamp values between 0 and 1
    s = 0.000000001
    this_j = (torch.sum(AnB) + s) / (torch.sum(AuB) + s)  # Compute Jaccard Index
    return this_j

def cal_dice(true, predict):
    # Dice coefficient
    intersection = torch.sum(true * predict)
    union = torch.sum(true) + torch.sum(predict)
    s = 0.000000001
    dice = (2. * intersection + s) / (union + s)
    return dice

def cal_ap_video(true, predict):
    # Move tensors to CPU before conversion
    true_cpu = true.cpu().numpy()
    predict_cpu = predict.cpu().numpy()
    ap = accuracy_score(true_cpu, predict_cpu)
    return ap 

def cal_ap_frame(true, predict):
    average_precision_frame = []

    for i in range(len(true[0])):
        ap_frame = accuracy_score(true[:, i].cpu().numpy(), predict[:, i].cpu().numpy())
        average_precision_frame.append(ap_frame)

    return average_precision_frame

def cal_fpr(true, predict, class_indices):
    false_positives = 0
    true_negatives = 0
    true_np = true.cpu().numpy()
    predict_np = predict.cpu().numpy()

    for class_idx in class_indices:
        true_class = true_np[class_idx]
        predict_class = predict_np[class_idx]
        false_positives += np.sum((predict_class == 1) & (true_class == 0))
        true_negatives += np.sum((predict_class == 0) & (true_class == 0))

    fpr = false_positives / (false_positives + true_negatives + 1e-10)  # Adding a small value to avoid division by zero
    return fpr

def cal_tnr(true, predict, class_indices):
    true_negatives = 0
    false_positives = 0
    true_np = true.cpu().numpy()
    predict_np = predict.cpu().numpy()

    for class_idx in class_indices:
        true_class = true_np[class_idx]
        predict_class = predict_np[class_idx]
        true_negatives += np.sum((predict_class == 0) & (true_class == 0))
        false_positives += np.sum((predict_class == 1) & (true_class == 0))

    tnr = true_negatives / (true_negatives + false_positives + 1e-10)  # Adding a small value to avoid division by zero
    return tnr

def cal_all_metrics(read_id,Output_root, label_mask, frame_label, video_label, predic_mask_3D, output_video_label,output_frame_label):
    device = label_mask.device
    predic_mask_3D=predic_mask_3D.to(device)
    output_video_label =output_video_label.to(device)
    ch, D, H, W = label_mask.size()
    predic_mask_3D =   F.interpolate(predic_mask_3D,  size=( H, W), mode='bilinear', align_corners=False)
    predic_mask_3D= (predic_mask_3D>0)*predic_mask_3D
    predic_mask_3D = predic_mask_3D -torch.min(predic_mask_3D)
    predic_mask_3D = predic_mask_3D /(torch.max(predic_mask_3D)+0.0000001)*1 
    predic_mask_3D = predic_mask_3D>0.1
    predic_mask_3D = torch.clamp(predic_mask_3D,0,1)

    output_video_label = (output_video_label > 0.5) * 1

    output_video_label_expanded = output_video_label.reshape(ch, 1, 1, 1) 
    output_video_label_expanded = output_video_label_expanded.repeat(1, D, H, W)
    predic_mask_3D = predic_mask_3D * output_video_label_expanded

    frame_label = frame_label.permute(1, 0)
    # Sum along the spatial dimensions to get frame-level predictions
    predic_frame = torch.sum(predic_mask_3D, dim=(-1, -2))
    # Sum along the frame dimension to get video-level predictions
    predic_video_from_cam = torch.max(predic_frame, dim=(-1))[0]
    predic_video_from_cam = (predic_video_from_cam > 1000) * 1
    # Calculate video-level AP from camera
    video_ap_from_cam = cal_ap_video(video_label, predic_video_from_cam)
    print("Video AP from cam:", video_ap_from_cam)
    
    # Calculate video-level AP from model output
    video_ap = cal_ap_video(video_label, output_video_label)
    print("Video AP from model output:", video_ap)
    predic_frame = (predic_frame > 100) * 1
    if output_frame_label is not None:
        predic_frame = (output_frame_label[0] > 0.5) * 1



    frame_ap = cal_ap_frame(frame_label, predic_frame)
    print("Frame AP from model output:", frame_ap)

    # Calculate Intersection over Union (IoU) for video-level predictions
    IoU = cal_J(label_mask[0], predic_mask_3D[0])
    IoU = round(IoU.item(), 4)
    print("Intersection over Union (IoU):", IoU)

    # Calculate Dice coefficient for video-level predictions
    dice = cal_dice(label_mask, predic_mask_3D)
    dice = round(dice.item(), 4)
    print("Dice coefficient:", dice)

    # Apply threshold to frame predictions
    # predic_frame = (predic_frame > 20) * 1.0
    
    # Expand the dimensions of predic_frame to match the shape of label_mask
    predic_frame_expanded = predic_frame.reshape(ch, D, 1, 1) 
    predic_frame_expanded = predic_frame_expanded.repeat(1, 1, H, W)
    # Calculate IoU for positive frames only
    IoU_maskout = cal_J(label_mask * predic_frame_expanded, predic_mask_3D * predic_frame_expanded)
    IoU_maskout = round(IoU_maskout.item(), 4)
    print("IoU for positive frames only:", IoU_maskout)

    # Calculate Dice coefficient for positive frames only
    Dice_maskout = cal_dice(label_mask * predic_frame_expanded, predic_mask_3D * predic_frame_expanded)
    Dice_maskout = round(Dice_maskout.item(), 4)
    print("Dice coefficient for positive frames only:", Dice_maskout)
     # Calculate FPR for frame-level predictions
    selected_class_indices = [0, 2]
    frame_fpr = cal_fpr(frame_label, predic_frame,selected_class_indices)
    print("Frame FPR from model output:", frame_fpr)

    # Calculate TNR for selected classes in frame-level predictions
    selected_class_indices = [0,1,2,3,4,5,6]
    frame_tnr = cal_tnr(frame_label, predic_frame, selected_class_indices)
    print("Frame TNR for selected classes:", frame_tnr)
    # Add data to metrics_video_data
    # Calculate average Hausdorff distance for all 29 frames in the first channel
    selected_class_indices = [0, 2]

    avg_hausdorff_first_channel = cal_hausdorff(label_mask* predic_frame_expanded, predic_mask_3D* predic_frame_expanded,selected_class_indices)
    avg_hausdorff_first_channel = round(avg_hausdorff_first_channel, 4)
    print("Average Hausdorff distance for first channel:", avg_hausdorff_first_channel)

    global metrics_video_data
    metrics_video_data.append({'read_id': read_id, 'Video_AP_Cam': video_ap_from_cam, 'Video_AP_Model': video_ap, 'IoU': IoU, 'IoU_Positive_Frames': 
                               IoU_maskout, 'Dice_Coefficient': dice, 'Dice_Coefficient_Positive_Frames':  Dice_maskout,
                                 'Frame_FPR': frame_fpr, 'Avg_Frame_TNR': frame_tnr,'Avg_Hausdorff_First_Channel': avg_hausdorff_first_channel})

    # Add frame-level accuracy scores to metrics_frame_data
    global metrics_frame_data
    new_frame_data = {'read_id': read_id}
    for i in range(len(frame_ap)):
        new_frame_data[f'Frame_{i+1}_AP'] = frame_ap[i]
    metrics_frame_data.append(new_frame_data)

    # Convert lists to DataFrames
    metrics_video = pd.DataFrame(metrics_video_data)
    metrics_frame = pd.DataFrame(metrics_frame_data)

    # Save to Excel files
    metrics_video.to_excel(Output_root+'metrics_video.xlsx', index=False, float_format='%.4f')
    metrics_frame.to_excel(Output_root+'metrics_frame.xlsx', index=False, float_format='%.4f')

# Example usage
# cal_all_metrics(...)
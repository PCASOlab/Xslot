
# the model
import cv2
import numpy

import os
import shutil
# from train_display import *
# the model
# import arg_parse
from visdom import Visdom
import random
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import   model_infer_slot_att
from working_dir_root import Output_root,Save_flag,Display_visdom_figure,Load_flow,Test_on_cholec_seg8k,Display_images, Evaluation
from dataset.dataset import myDataloader,categories,category_colors, Endovis_data, MICCAI_data
from working_dir_root import Evaluation_slots
from dataset import io
from working_dir_root import selected_data, Visdom_flag
import eval
import eval_box
import eval_slots
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if Visdom_flag:
    viz = Visdom(port=8097)
show_num = 10
# Save_flag =False
def save_img_to_folder(this_save_dir,ID,img):
    # this_save_dir = Output_root + "1out_img/" + Model_key + "/ground_circ/"
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)
    cv2.imwrite(this_save_dir +
                str(ID) + ".jpg", img)

class Display(object):
    def __init__(self,parser=None):
        self.Model_infer = model_infer_slot_att._Model_infer(args=parser)
        self.dataLoader = myDataloader()
        self.show_num=show_num
        self.results = {"score":[],
                        "count":[],
                        "cpr_cont":[]
                        }
        

    def train_display(self,MODEL_infer,mydata_loader, read_id,Output_root):
        # copy all the input videos and labels
        # cv2.destroyAllWindows()
        if type(MODEL_infer.final_output) is list and MODEL_infer.final_output is not None:
            self.Model_infer.output= MODEL_infer.final_output[0]

            print("It's a list!")
        else:
            self.Model_infer.output= MODEL_infer.final_output

            print("It's not a list.")
        # self.Model_infer.slice_valid = MODEL_infer.slice_valid
        self.Model_infer.cam3D = MODEL_infer.cam3D
        self.Model_infer.raw_cam = MODEL_infer.raw_cam

        self.dataLoader.input_videos = mydata_loader.input_videos
        self.dataLoader.labels = mydata_loader.labels
        self.dataLoader.input_flows = mydata_loader.input_flows
        # self.Model_infer.input_resample = MODEL_infer.input_resample
        self.dataLoader.all_raw_labels = mydata_loader.all_raw_labels
        self.dataLoader.this_file_name = mydata_loader.this_file_name

        self.Model_infer.direct_frame_output  = MODEL_infer.direct_frame_output

        if (Test_on_cholec_seg8k or Endovis_data) and Evaluation:
            self.dataLoader.this_label_mask  = mydata_loader.this_label_mask
            self.dataLoader.this_frame_label = mydata_loader.this_frame_label
            self.dataLoader.this_video_label = mydata_loader.this_video_label

            label_mask = torch.from_numpy(np.float32(self.dataLoader.this_label_mask )).to (device)
            frame_label = torch.from_numpy(np.float32(self.dataLoader.this_frame_label )).to (device)
            video_label = torch.from_numpy(np.float32(self.dataLoader.this_video_label )).to (device)


            # self.Model_infer.cam3D[0,2:7,:,:]*=0
            # label_mask[2:7,:,:]*=0
            eval.cal_all_metrics(read_id,Output_root,label_mask,frame_label,video_label, 
                                 self.Model_infer.cam3D[0],self.Model_infer.output[0,:,0,0,0].detach(),self.Model_infer.direct_frame_output)
        if (selected_data == "YTOBJ" and Evaluation):
            self.dataLoader.this_frame_label = mydata_loader.this_frame_label
            self.dataLoader.this_video_label = mydata_loader.this_video_label
            self.dataLoader.this_boxs  = mydata_loader.this_boxs
            self.dataLoader.this_masks = mydata_loader.this_masks
            frame_label = torch.from_numpy(np.float32(self.dataLoader.this_frame_label )).to (device)
            video_label = torch.from_numpy(np.float32(self.dataLoader.this_video_label )).to (device)
            bz,_,_,_,_ =  self.Model_infer.cam3D.shape
            self.result_corr=eval_box.cal_all_metrics_box(read_id,Output_root,self.dataLoader.this_masks,frame_label,video_label, 
                                 self.Model_infer.cam3D[bz-1],self.Model_infer.output[bz-1,:,0,0,0].detach(),self.Model_infer.direct_frame_output,self.dataLoader.input_videos[bz-1],self.results)
            # print("iou" + str(this_iou))z-1# print("iou" + str(this_iou))
            # self.Model_infer.cam3D[0] = label_mask
        if (Evaluation_slots == True):
            self.dataLoader.this_label_mask  = mydata_loader.this_label_mask
            self.dataLoader.this_frame_label = mydata_loader.this_frame_label
            self.dataLoader.this_video_label = mydata_loader.this_video_label
            bz,_,_,_,_ =  self.Model_infer.cam3D.shape

            label_mask = torch.from_numpy(np.float32(self.dataLoader.this_label_mask )).to (device)
            frame_label = torch.from_numpy(np.float32(self.dataLoader.this_frame_label )).to (device)
            video_label = torch.from_numpy(np.float32(self.dataLoader.this_video_label )).to (device)

            # (read_id, Output_root, label_mask, predic_mask_3D, output_video_label):
            self.result_corr=eval_slots.cal_all_metrics_slots(read_id,Output_root,label_mask,self.Model_infer.cam3D[0],self.dataLoader.input_videos[bz-1])

        if hasattr(MODEL_infer, 'sam_mask'):
            self.Model_infer. sam_mask =  MODEL_infer.sam_mask
             
        else:
            print("Parameter sam mask does not exist or is NaN")
        if Load_flow == True:
            Gray_video = self.dataLoader.input_flows[0,:,:,:] # RGB together
            Ori_D,Ori_H,Ori_W = Gray_video.shape
            step_l = int(Ori_D/self.show_num)+1
            for i in range(0,Ori_D,step_l):
                if i ==0:
                    stack = Gray_video[i]
                else:
                    stack = np.hstack((stack,Gray_video[i]))

            # Display the final image
            # cv2.imshow('Stitched in put flows', stack.astype((np.uint8)))
            # cv2.waitKey(1)


        # Gray_video = self.Model_infer.input_resample[0,2,:,:,:].cpu().detach().numpy()# RGB together
            ### OG video #################################
        Gray_video = self.dataLoader.input_videos[0,:,:,:,:] # RGB together
        ch,Ori_D,Ori_H,Ori_W = Gray_video.shape
        Gray_video = np.transpose(Gray_video,(1,2,3,0))
        step_l = int(Ori_D/self.show_num)+1
        for i in range(0,Ori_D,step_l):
            if i ==0:
                stack1 =  Gray_video[i] 
            else:
                stack1 = np.hstack((stack1,Gray_video[i]))
        # stack1 = np.array(cv2.merge((stack1, stack1, stack1)))

        # Display the final image
        # cv2.imshow('Stitched in put Image', stack1.astype((np.uint8)))
        # cv2.waitKey(1)

        if Save_flag == True:
            io.save_img_to_folder(Output_root + "image/original/" ,  read_id, stack1.astype((np.uint8)) )
        
        # Combine the rows vertically to create the final 3x3 arrangement
        Cam3D=self.Model_infer.raw_cam[0]
        final_mask = self.Model_infer.cam3D[0].cpu().detach().numpy()
        label_0 = self.dataLoader.labels[0]
        if len (Cam3D.shape) == 3:
            Cam3D = Cam3D.unsqueeze(1)
        ch, D, H, W = Cam3D.size()
        

        # activation = nn.Sigmoid()
        # Cam3D =  activation( Cam3D)
        # average_tensor = Cam3D.mean(dim=[1,2,3], keepdim=True)
        # _, sorted_indices = average_tensor.sort(dim=0)
        if len (self.Model_infer.output.shape) == 5:
            output_0 = self.Model_infer.output[0,:,0,0,0].cpu().detach().numpy()
        else:
            output_0 = self.Model_infer.output[0,:,0,0].cpu().detach().numpy()
        step_l = int(D/self.show_num)+1
        stitch_i =0
        stitch_im  = np.zeros((H,W))
        stitch_over = np.zeros((H,W))
        # ch, D, H_m, W_m = final_mask.shape
        # color_mask = np.zeros((D,H_m,W_m,3))
        # stack_color_mask = np.zeros((H,W))
        Min_len = np.min((len(Cam3D),len(label_0)))
        for j in range(Min_len):
            # j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
            this_grayVideo = Cam3D[j].cpu().detach().numpy()
            # this_mask_channel = final_mask[j].cpu().detach().numpy()
            if (output_0[j]>0.5 or label_0[j]>0.5):
                for i in range(0, D, step_l):
                    this_image = this_grayVideo[i]
                    this_image =  cv2.resize(this_image, (Ori_H, Ori_W), interpolation = cv2.INTER_LINEAR)
                     
                    if i == 0:
                        stack = this_image
                        
                    else:
                        stack = np.hstack((stack, this_image))
                        
                stack= (stack>0)*stack
                stack = stack -np.min(stack)
                stack = stack /(np.max(stack)+0.0000001)*254 
                # stack
                # stack = (stack>20)*stack
                # stack = (stack>0.5)*128
                stack = np.clip(stack,0,254)
                stack = cv2.applyColorMap(stack.astype((np.uint8)), cv2.COLORMAP_JET)
                # stack = cv2.merge((stack, stack, stack))

                alpha= 0.6
                overlay = cv2.addWeighted(stack1.astype((np.uint8)), 1 - alpha, stack.astype((np.uint8)), alpha, 0)
                # overlay = np.clip(overlay,0,254)

                # stack =  stack - np.min(stack)
                infor_image = this_image*0
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                font_color = (255, 255, 255)  # White color

                
                if j < len (categories):
                    text1 = str(j) + "S"+ "{:.2f}".format(output_0[j])  
                
                    text2="G"+ str(label_0[j])
                    text3 = categories[j]
                else:
                    text1=text2=text3 =  "Nan"
                # Define the position where you want to put the text (bottom-left corner)
                text_position = (5, 20)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text1, text_position, font, font_scale, font_color, font_thickness)
                text_position = (5, 30)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text2, text_position, font, font_scale, font_color, font_thickness)
                text_position = (5, 40)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text3, text_position, font, font_scale, font_color, font_thickness)
                # stack = stack -np.min(stack)
                # stack = stack /(np.max(stack)+0.0000001)*254
                infor_image = cv2.merge((infor_image, infor_image, infor_image))

                stack = np.hstack((infor_image, stack))
                overlay = np.hstack((infor_image, overlay))
               
                # Display the final image
                # cv2.imshow( str(j) + "score"+ "{:.2f}".format(output_0[j]) + "GT"+ str(label_0[j])+categories[j], stack.astype((np.uint8)))
                # cv2.waitKey(1)
                if stitch_i ==0:
                    stitch_im = stack
                    stitch_over = overlay
                else:
                    stitch_im = np.vstack((stitch_im, stack))
                    stitch_over = np.vstack((stitch_over, overlay))

                stitch_i+=1
        
        stack_color_mask=stack_to_color_mask (final_mask,Ori_H,Ori_W,output_0,label_0,step_l)
        if stack_color_mask is not None:
            alpha= 0.5
            stack_color_mask = cv2.addWeighted(stack1.astype((np.uint8)), 1 - alpha, stack_color_mask.astype((np.uint8)), alpha, 0)
            if Save_flag == True:
                # masks after threshold
                io.save_img_to_folder(Output_root + "image/predict_color_mask/" ,  read_id, stack_color_mask.astype((np.uint8)) )
            if Display_visdom_figure:

                stack_color_mask =  cv2.cvtColor(stack_color_mask, cv2.COLOR_RGB2BGR)
                viz.image(np.transpose(stack_color_mask.astype((np.uint8)), (2, 0, 1)), opts=dict(title=f'{read_id} - stack_color_mask'))

        if   (Evaluation or Evaluation_slots):
            final_mask = label_mask.cpu().detach().numpy()
            gt_color_mask=stack_to_color_mask (final_mask,Ori_H,Ori_W,label_0,output_0,step_l)
            if gt_color_mask is not None:

                alpha= 0.5
                gt_color_mask = cv2.addWeighted(stack1.astype((np.uint8)), 1 - alpha, gt_color_mask.astype((np.uint8)), alpha, 0)
                if Save_flag == True:
                    io.save_img_to_folder(Output_root + "image/GT_color_mask/" ,  read_id, gt_color_mask.astype((np.uint8)) )
        # for j in range(len(categories)):
        #     # j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
           
        #     this_mask_channel = final_mask[j]
        #     color_mask[this_mask_channel > 0.5] = category_colors[categories[j]]
        #     if (output_0[j]>0.5 or label_0[j]>0.5):
        #         for i in range(0, D, step_l):
                    
        #             this_mask_channel_frame = color_mask[i]
        #             this_mask_channel_frame =  cv2.resize(this_mask_channel_frame, (Ori_H, Ori_W), interpolation = cv2.INTER_LINEAR)
                     
        #             if i == 0:
                      
        #                 stack_color_mask = this_mask_channel_frame
        #             else:
                         
        #                 stack_color_mask = np.hstack((stack_color_mask, this_mask_channel_frame))
        image_all = np.vstack((stitch_over,stitch_im))
        if Display_images:
            cv2.imshow( 'all', image_all.astype((np.uint8)))
            # cv2.imshow( 'overlay', stitch_over.astype((np.uint8)))

            cv2.waitKey(1)
        if Save_flag == True:

            io.save_img_to_folder(Output_root + "image/predict/" ,  read_id, stitch_over.astype((np.uint8)) )
            io.save_img_to_folder(Output_root + "image/predict_overlay/" ,  read_id, image_all.astype((np.uint8)) )
        if Display_visdom_figure:

            stitch_over =  cv2.cvtColor(stitch_over, cv2.COLOR_RGB2BGR)
            # viz.image(np.transpose(stitch_over.astype((np.uint8)), (2, 0, 1)), opts=dict(title=f'{read_id} - predict_overlay'))




        if MODEL_infer.gradcam is not None:
            heatmap = MODEL_infer.gradcam[0,0,:,:].cpu().detach().numpy()

            # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-5)

                # Resize the heatmap to the original image size
            # heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

            # Apply colormap to the heatmap
            heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # Superimpose the heatmap on the original image
            # result = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.7, heatmap_colormap, 0.3, 0)

            # Display the result
            cv2.imshow('Grad-CAM', heatmap_colormap)
            cv2.waitKey(1)
        # Cam3D = nn.functional.interpolate(side_out_low, size=(1, Path_length), mode='bilinear')
# Helper function to generate a random color
def generate_random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

def stack_to_color_mask(final_mask, Ori_H, Ori_W, output_0, label_0, step_l):
    ch, D, H_m, W_m = final_mask.shape
    color_mask = np.zeros((D, H_m, W_m, 3))  # 4D array to store color masks (D, H, W, 3)

    # Dictionary to store dynamically generated colors if we exceed category_colors
    additional_colors = {}

    stack_color_mask = None
    for j in range(len(final_mask)):
        # this_mask_channel is the mask for the current channel
        this_mask_channel = final_mask[j]

        # If j exceeds the number of predefined category colors, generate a new color
        if j >= len(category_colors):
            if j not in additional_colors:
                # Generate and store a new random color for this index
                additional_colors[j] = generate_random_color()
            color = additional_colors[j]
        else:
            # Use predefined category color
            color = list(category_colors.values())[j]

        # Apply the color to the mask where this_mask_channel > 0.3
        color_mask[this_mask_channel > 0.3] = color

        # Only proceed if output_0 or label_0 for this channel is greater than 0.5
        if output_0[j] > 0.5 or label_0[j] > 0.5:
            for i in range(0, D, step_l):
                # Resize the mask to the original dimensions
                this_mask_channel_frame = color_mask[i]
                this_mask_channel_frame = cv2.resize(this_mask_channel_frame, (Ori_H, Ori_W), interpolation=cv2.INTER_LINEAR)

                # Stack the resized masks horizontally
                if i == 0:
                    stack_color_mask = this_mask_channel_frame
                else:
                    stack_color_mask = np.hstack((stack_color_mask, this_mask_channel_frame))

    return stack_color_mask
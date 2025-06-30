#THe data set read the PKL file for Contour with sheath so the contact can be detected
import cv2
import numpy as np
import csv
import re
import os
print("Current working directory:", os.getcwd())
from time import  time
import pandas as pd
import dataset.io as io
import random
import image_operator.basic_operator as basic_operator
import torch
from dataset import format_convertor

from working_dir_root import Dataset_video_root, Dataset_label_root, Dataset_video_pkl_merge_root,Dataset_video_pkl_flow_root,Batch_size,Random_mask
from working_dir_root import Dataset_video_pkl_cholec,Random_Full_mask,output_folder_sam_feature,Data_aug,train_test_list_dir
from working_dir_root import Test_on_cholec_seg8k, Dataset_video_pkl_cholec8k, output_folder_sam_feature_cholec8k
from working_dir_root import Dataset_video_pkl_thoracic,Dataset_video_pkl_endovis,Dataset_video_pkl_Davis,Dataset_video_pkl_YTVOS,Dataset_video_pkl_MICCAIv2
from working_dir_root import sam_feature_OLG_dir,selected_data,data_flag_list,Dataset_video_pkl_root,sam_feature_OLG_dir2,sam_feature_OLG_dir3
from working_dir_root import Davis_categories_list,Davis_super_category_list,YTVOS_categories_list,YTOBJ_categories_list
from working_dir_root import Dataset_video_pkl_YTOBJ,Crop_half,Evaluation,Video_len,Load_prototype, Slots_prototype_pkl,Data_percentage
from working_dir_root import Slots_label_dir, Load_prototype_label, Evaluation_slots, Dataset_video_pkl_MICCAI_test,working_pcaso_raid
from working_dir_root import Thoracic_select,Video_down_sample_f,Dataset_video_pkl_thoracic_test
# data_flag_list = ["Cholec_data_flag", "Thoracic_data_flag", "MICCAI_data", "Endovis_data"]


if Test_on_cholec_seg8k == True:
   output_folder_sam_feature=  output_folder_sam_feature_cholec8k
   Dataset_video_pkl_cholec = Dataset_video_pkl_cholec8k
Seperate_LR = False
Mask_out_partial_label = False
input_ch = 3 # input channel of each image/video

# selected_data  =  "Thoracic_data_flag"
Cholec_data_flag = False
Thoracic_data_flag = False
MICCAI_data = False
Endovis_data = False
MICCAI_data_merge = False
DAVIS_data= False
YTVOS_data = False
YTOBJ_data = False
Cholec_data_flag = "Cholec" in selected_data
Thoracic_data_flag = "Thoracic" in selected_data 
MICCAI_data = "MICCAI" in selected_data
MICCAI_data_merge = "MICCAI_merge" in selected_data
MICCAIv2 = "MICCAIv2" in selected_data
Endovis_data = "Endovis" in selected_data
DAVIS_data = "DAVIS" in selected_data
YTVOS_data = "YTVOS" in selected_data
YTOBJ_data = "YTOBJ" in selected_data

# Cholec_data_flag,Thoracic_data_flag,MICCAI_data,MICCAI_data_merge,MICCAIv2,Endovis_data,DAVIS_data, YTVOS_data,YTOBJ_data= [data_flag_list[0] == selected_data,data_flag_list[1] == selected_data, 
#                                     data_flag_list[2] == selected_data, data_flag_list[3] == selected_data, data_flag_list[4] == selected_data,
#                                     data_flag_list[5] ==selected_data, data_flag_list[6] ==selected_data,data_flag_list[7] ==selected_data ,data_flag_list[8] ==selected_data ]

# Validation logic
flags= [Cholec_data_flag,Thoracic_data_flag,MICCAI_data,MICCAI_data_merge,MICCAIv2,Endovis_data,DAVIS_data, YTVOS_data,YTOBJ_data]
if not any(flags):
    # If none of the flags are True, raise an exception
    raise ValueError("At least one flag must be True.")
elif sum(flags) > 1:
    print("Mixed training data")
    # If more than one flag is True, raise an exception
    # raise ValueError("Only one flag can be True.")

categories_count = [17, 13163, 17440, 576, 1698, 4413, 11924, 10142, 866, 2992,131, 17, 181, 1026]

total_samples = sum(categories_count)
class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]

# weight_tensor = torch.tensor(class_weights, dtype=torch.float)
Select_on_label = False
removed_category =[0,3,5,8,10,11,12,13]
categories = [
    'bipolar dissector', #0   - 17
    'bipolar forceps', #1     -13163
    'cadiere forceps', #2     -17440
    'clip applier', #3        -576
    'force bipolar',#4         - 1698
    'grasping retractor',#5     -4413
    'monopolar curved scissors',#6     -11924
    'needle driver', #7                 -10142
    'permanent cautery hook/spatula', # 8     -866
    'prograsp forceps', #9                -2992
    'stapler', #10                      -131
    'suction irrigator', #11             -17
    'tip-up fenestrated grasper', #12       -181
    'vessel sealer' #13                  -1026
]
category_colors = {
    'bipolar dissector': (255, 0, 0),    # Red
    'bipolar forceps': (0, 255, 0),      # Green
    'cadiere forceps': (0, 0, 255),      # Blue
    'clip applier': (255, 128, 0),       # Orange
    'force bipolar': (128, 0, 128),     # Purple
    'grasping retractor': (0, 255, 255), # Cyan
    'monopolar curved scissors': (255, 128, 128), # Light Red
    'needle driver': (128, 128, 0),     # Olive
    'permanent cautery hook/spatula': (128, 0, 255), # Indigo
    'prograsp forceps': (0, 128, 128),  # Teal
    'stapler': (255, 0, 128),           # Pink
    'suction irrigator': (128, 255, 0), # Lime
    'tip-up fenestrated grasper': (255, 128, 0),    # Light Orange
    'vessel sealer': (0, 128, 255)      # Light Blue
}

if Cholec_data_flag == True:
    categories = [
        'Grasper', #0   
        'Bipolar', #1    
        'Hook', #2    
        'Scissors', #3      
        'Clipper',#4       
        'Irrigator',#5    
        'SpecimenBag',#6                  
    ]
    category_colors = {
    'Grasper': (0, 0, 255),        # Blue
    'Bipolar': (0, 255, 0),        # Green
    'Hook': (255, 0, 0),           # Red
    'Scissors': (255, 255, 0),     # Yellow
    'Clipper': (255, 0, 255),      # Magenta
    'Irrigator': (255, 165, 0),    # Orange
    'SpecimenBag': (128, 0, 128)   # Purple
    }
    categories_count =[5266.0,  592.0, 4252.0,  239.0,  352.0,  624.0,  623.0]
    total_samples = sum(categories_count)
    class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]
    
if Thoracic_data_flag == True:
    categories = [
    'Lymph node',
    'Vagus nereve',
    'Bronchus',
    'Lung parenchyma',
    'Instruments', 
    ]
    category_colors = {
    'Lymph node': (0, 0, 255),        # Red
    'Vagus nereve': (0, 255, 0),      # Green
    'Bronchus': (255, 0, 0),          # Blue
    'Lung parenchyma': (255, 255, 0),  # Yellow
    'Instruments': (255, 0, 255),      # Magenta
    }
   
    categories_count =[575.0, 189.0, 427.0, 1651.0,  1928.0]
    total_samples = sum(categories_count)
    class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]

if Endovis_data == True:
    categories =  [
    'Prograsp_Forceps_labels',
    'Large_Needle_Driver_labels',
    'Grasping_Retractor_labels',
    'Bipolar_Forceps_labels',
    'Vessel_Sealer_labels',
    'Monopolar_Curved_Scissors_labels',
    'Other_labels'
    ]
        
    category_colors = {
        'Prograsp_Forceps_labels': (255, 0, 0),            # Red
        'Large_Needle_Driver_labels': (0, 255, 0),         # Green
        'Grasping_Retractor_labels': (0, 0, 255),          # Blue
        'Bipolar_Forceps_labels': (0, 255, 255),           # Cyan
        'Vessel_Sealer_labels': (255, 0, 255),             # Magenta
        'Monopolar_Curved_Scissors_labels': (128, 0, 128), # Purple
        'Other_labels': (255, 165, 0),                     # Gray
    }
    categories_count =[5.0,  5.0, 5.0,  5.0,  5.0,  5.0,  5.0,5.0, 5.0,5.0]
    total_samples = sum(categories_count)
    class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]
if DAVIS_data:
    categories= list(Davis_categories_list.keys())
    # categories= list(Davis_super_category_list )

    
    categories_count = np.ones(len(categories))
    total_samples = sum(categories_count)
    class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]
if YTVOS_data:
    categories= list(YTVOS_categories_list)
    # categories= list(Davis_super_category_list )
 
    categories_count = np.ones(len(categories))
    total_samples = sum(categories_count)
    class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]
if YTOBJ_data:
    categories= list(YTOBJ_categories_list )
    # categories= list(Davis_super_category_list )
 
    categories_count = np.ones(len(categories))
    total_samples = sum(categories_count)
    class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]

Obj_num = len(categories)
if Mask_out_partial_label == True:
    label_mask = np.zeros(Obj_num)
    label_mask[4]=1
else:
    label_mask = np.ones(Obj_num)

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def convert_to_one_hot(tool, categories):
    vector = [0] * (len(categories) + 1)  # Initialize the vector with zeros
    if tool == 'b':  # Background
        vector[-1] = 1  # Set the last digit for background
    else:
        if tool in categories:
            index = categories.index(tool)
            vector[index] = 1
    return vector
class myDataloader(object):
    def __init__(self, OLG=False,img_size = 128,Display_loading_video = False,
                 Read_from_pkl= True,Save_pkl = False,Load_flow =False,Load_feature=True,Train_list = "else",Device=device):
        print("GPU function is : "+ str(cv2.cuda.getCudaEnabledDeviceCount()))
        self.device = Device
        self.categories = categories
        self.categories_count = categories_count
        self.image_size = img_size
        self.Display_loading_video =Display_loading_video
        self.Read_from_pkl= Read_from_pkl 
        self.Save_pkl=Save_pkl
        self.batch_size = Batch_size
        self.obj_num = Obj_num
        self.Load_flow=Load_flow
        self.Load_feature = Load_feature
        self.video_down_sample = 60  # 60 FPS
        self.cut_video_len =Video_len # -1 the longest   
        self.video_len = 5
        self.video_buff_size = int(60/self.video_down_sample) * self.video_len  # each video has 30s discard last one for flow
        self.OLG_flag = OLG
        self.create_train_list= True
        self.GT = True
        self.dataset_percentage = Data_percentage
        self.noisyflag = False
        self.Random_rotate = True
        self.Random_vertical_shift = True
        self.input_images= np.zeros((self.batch_size, 1, img_size, img_size))
        # self.input_videos = np.zeros((self.batch_size,3,self.video_buff_size,img_size,img_size )) # RGB together
        # switch to smart batch 
        self.input_videos =[]
        self.input_flows = np.zeros((self.batch_size,self.video_buff_size,img_size,img_size )) # RGB together
        self.features =[]
        # the number of the contour has been increased, and another vector has beeen added
        self.labels_LR= np.zeros((self.batch_size,2*self.obj_num))  # predifine the path number is 2 to seperate Left and right
        self.labels= np.zeros((self.batch_size, self.obj_num))  # left right merge
        self.features_exist = True

        self.all_read_flag =0
        self.save_id =0
        self.read_record = 0
        self.all_video_dir_list = []
        if   MICCAI_data == True or MICCAI_data_merge == True:
            self.all_labels = self.load_all_lables()
        if Read_from_pkl == False:
            self.all_video_dir_list = os.listdir(Dataset_video_root)
        else:
            if MICCAI_data == True:
                full_list_pkl = os.listdir(Dataset_video_pkl_root)
                if Evaluation_slots ==True:
                    files = os.listdir(Dataset_video_pkl_MICCAI_test)
                    self.all_video_dir_list.extend([(f, 'miccai') for f in files])
                else:
                    record_list = io.read_a_pkl(working_pcaso_raid + "MICCAI_selected_GT/", 'unselected_videos')
                    record_list_pkl = [file.replace('.mp4', '.pkl') for file in record_list]
                    # Create a list of overlap between full_list_pkl and the updated record_list_pkl
                    files = list(set(full_list_pkl) & set(record_list_pkl))  # 24541
                    self.all_video_dir_list.extend([(f, 'miccai') for f in files])
            if MICCAI_data_merge == True:
                # Dataset_video_pkl_root = Dataset_video_pkl_merge_root
                self.all_video_dir_list = os.listdir(Dataset_video_pkl_merge_root)
            if MICCAIv2 == True:
                self.all_video_dir_list = os.listdir(Dataset_video_pkl_MICCAIv2)


            if Cholec_data_flag == True:
                files = os.listdir(Dataset_video_pkl_cholec)
                self.all_video_dir_list.extend([(f, 'cholec') for f in files])
                


            if Thoracic_data_flag == True:
                
                if Evaluation_slots == True:
                    files= os.listdir(Dataset_video_pkl_thoracic_test)
                    self.all_video_dir_list.extend([(f, 'thoracic') for f in files])
                else:
                    files  = os.listdir(Dataset_video_pkl_thoracic)
                    self.all_video_dir_list.extend([(f, 'thoracic') for f in files])
                # if Evaluation_slots ==True:
                #     record_list = io.read_a_pkl(Thoracic_select, 'selected_videos')
                # else:
                #     record_list = io.read_a_pkl(Thoracic_select, 'unselected_videos')
                # self.all_video_dir_list =  record_list  # 24541
            if Endovis_data == True:
                files = os.listdir(Dataset_video_pkl_endovis)
                self.all_video_dir_list.extend([(f, 'endovis') for f in files])
            if DAVIS_data:
                self.all_video_dir_list = os.listdir(Dataset_video_pkl_Davis)
            if YTVOS_data:
                self.all_video_dir_list = os.listdir (Dataset_video_pkl_YTVOS)
            if YTOBJ_data:
                self.all_video_dir_list = os.listdir (Dataset_video_pkl_YTOBJ)

            if Train_list == "train":
                # train_set_no8k
                # self.all_video_dir_list = io.read_a_pkl(train_test_list_dir, 'train_set')
                # self.all_video_dir_list = io.read_a_pkl(train_test_list_dir, 'train_set_no8k')
                self.all_video_dir_list = io.read_a_pkl(train_test_list_dir, 'train_set_balance_no8K')
            elif Train_list == "train_raw":
                self.all_video_dir_list = io.read_a_pkl(train_test_list_dir, 'train_set_no8k')
            elif Train_list == "small":
                self.all_video_dir_list = io.read_a_pkl(train_test_list_dir, 'train_set')
            elif Train_list == "train_set_partition":
                self.all_video_dir_list = io.read_a_pkl(train_test_list_dir, 'train_set_partition')
            
            if Test_on_cholec_seg8k ==True:
                files = os.listdir(Dataset_video_pkl_cholec8k)
                # files = os.listdir(Dataset_video_pkl_cholec)
                self.all_video_dir_list.extend([(f, 'cholec') for f in files])
        if Load_prototype_label:
            self.all_prototype_label_list = os.listdir (Slots_label_dir)
        self.all_video_dir_list = sorted(self.all_video_dir_list)
        if Evaluation == False and Evaluation_slots==False:
            random.shuffle(self.all_video_dir_list)
        if Evaluation_slots == False:
            num_items_to_keep = int(len(self.all_video_dir_list) * self.dataset_percentage)

        # Shorten the list
            self.all_video_dir_list = self.all_video_dir_list [:num_items_to_keep]
        self.video_num = len (self.all_video_dir_list)

        #Guiqiu modified for my computer
        # self.com_dir =  Generator_Contour_sheath().com_dir # this dir is for the OLG
        # if self.OLG_flag == True:
        #      # initial lizt the
        #     self.talker = Communicate()
    def load_all_lables(self): # load all labels and save then as dictionary format
        csv_file_path = Dataset_label_root + "labels.csv"
        if MICCAI_data_merge == True:
            csv_file_path = Dataset_label_root + "labels_merge.csv"

        # Initialize an empty list to store the data from the CSV file
        data = []
        sum = np.zeros(self.obj_num)
        # Open the CSV file and read its contents
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)

                # Read the header row (if any)
                header = next(csvreader)

                # Read the remaining rows and append them to the 'data' list
                for row in csvreader:
                    data.append(row)
                    binary_vector = np.array([1 if category in row[2] else 0 for category in categories], dtype=int)
                    sum = sum+ binary_vector
        except FileNotFoundError:
            print(f"File not found at path: {csv_file_path}")
            exit()
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()

        # Now you have the data from the CSV file in the 'data' list
        # You can manipulate or process the data as needed

        # Example: Printing the first few rows
        for row in data[:5]:
            print("all data is loaded and here are some samples:")
            print(row)

        labels = data
        # conver label list into dictionary that can used key for fast lookingup
        label_dict = {label_info[1]: label_info[2] for label_info in labels}  # use the full name as the dictionary key
        label_dict_number = {label_info[0]: label_info[2] for label_info in
                             labels}  # using the number and dictionary keey instead

        all_labels = label_dict
        return all_labels
    def convert_left_right_v(self,this_label):
        # label_element = re.findall(r'[^,]+|nan', this_label)  # change to vector format instead of string
        split_string = this_label.split(',', 2)
        this_label_l = ','.join(split_string[:2])
        this_label_r =  split_string[2]
        binary_vector_l = np.array([1 if category in this_label_l else 0 for category in categories], dtype=int)
        binary_vector_r = np.array([1 if category in this_label_r else 0 for category in categories], dtype=int)

        # readd
        return binary_vector_l, binary_vector_r
    def merge_labels(self,label_group):
        return np.max(label_group, axis=0)
    # load one video buffer (self.video_buff_size , 3, img_size, img_size),
    # and its squeesed which RGB are put together (self.video_buff_size * 3, img_size, img_size),
    def Align_stack_videos(self, video_list):
    # Determine minimum dimensions across all videos
        min_height = min(video.shape[2] for video in video_list)  # H dimension
        min_width = min(video.shape[3] for video in video_list)   # W dimension
        max_time = max(video.shape[1] for video in video_list)    # Temporal dimension
        
        # Get batch parameters
        N = len(video_list)
        C = 3  # Explicitly set for RGB
        
        # Initialize output tensor (N, C, T, H, W)
        Video_stack_aligned = np.zeros((N, C, max_time, min_height, min_width), dtype=np.uint8)
        
        # Process each video
        for i, video in enumerate(video_list):
            # Original dimensions: (C, T, H, W)
            _, T_orig, H_orig, W_orig = video.shape
            
            # Temporary array for resized video
            resized_video = np.zeros((C, T_orig, min_height, min_width), dtype=np.uint8)
            
            # Process each frame
            for t in range(T_orig):
                # Get full RGB frame and transpose to (H, W, C)
                frame = video[:, t].transpose(1, 2, 0)  # From (C,H,W) → (H,W,C)
                
                # Resize while maintaining color integrity
                resized_frame = cv2.resize(
                    frame, 
                    (min_width, min_height),  # Note: OpenCV uses (width, height)
                    interpolation=cv2.INTER_AREA
                )
                
                # Transpose back to (C, H, W) and store
                resized_video[:, t] = resized_frame.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
            
            # Pad temporal dimension if needed
            Video_stack_aligned[i, :, :T_orig] = resized_video
        
        # Apply temporal cropping if specified
        if self.cut_video_len != -1:
            Video_stack_aligned = Video_stack_aligned[:, :, :self.cut_video_len]
        
        return Video_stack_aligned
    
    def load_this_video_buffer(self,video_path ):
        cap = cv2.VideoCapture(video_path)

        # Read frames from the video clip
        flow_f_gap = 5
        frame_count = 0
        buffer_count = 0
        # Read frames from the video clip
        video_buffer = np.zeros((3,self.video_buff_size,  self.image_size, self.image_size))
        video_buffer2= np.zeros((3,self.video_buff_size,  self.image_size, self.image_size)) # neiboring buffer for flow
        flow_buffer= np.zeros((self.video_buff_size,  self.image_size, self.image_size)) # neiboring buffer for flow
        frame_number =0
        Valid_video=False
        this_frame = 0
        previous_frame = 0
        previous_count =0
        while True:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if ((frame_count % self.video_down_sample==0) or (frame_count == (previous_count+flow_f_gap))):
                # start_time = time()

                ret, frame = cap.read()
      
                if ret == True:
                    H, W, _ = frame.shape
                    crop = frame[0:H-80, 192:1088]
                    
                    if self.Display_loading_video == True:

                            cv2.imshow("crop", crop.astype((np.uint8)))
                            cv2.waitKey(1)
                    this_resize = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                    reshaped = np.transpose(this_resize, (2, 0, 1))


                    if frame_count % self.video_down_sample==0:
                        video_buffer[:, buffer_count, :, :] = reshaped
                        previous_count=frame_count
                    if frame_count == (previous_count+flow_f_gap):
                        video_buffer2[:, buffer_count, :, :] = reshaped


                        this_frame = video_buffer2[0, buffer_count, :, :]
                        previous_frame = video_buffer[0, buffer_count, :, :]
                        if self.Load_flow:
                            flow = cv2.calcOpticalFlowFarneback(
                                    previous_frame, this_frame, flow=None, pyr_scale=0.5, levels=3, winsize=5, iterations=8, poly_n=5, poly_sigma=1.1, flags=0
                                )

                            # Calculate magnitude of the flow vectors
                            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                            # Normalize and scale the magnitude to the range [0, 255]
                            magnitude = (magnitude - np.min(magnitude)) / np.max(magnitude) * 255
                            # magnitude = (magnitude - np.min(magnitude)) / np.max(magnitude) * 255

                            magnitude = np.clip(magnitude,0,254)
                            flow_buffer[ buffer_count, :, :] = magnitude
                        buffer_count += 1
                        if self.Display_loading_video == True:

                            cv2.imshow("crop", magnitude.astype((np.uint8)))
                            cv2.waitKey(1)
                   
                    if buffer_count >= self.video_buff_size:
                        buffer_count = 0
                        Valid_video =True
                        break
            else:
                ret = cap.grab()
                # counter += 1
            if not ret:
                break
            frame_count += 1
            frame_number +=1

        cap.release()
        # Squeeze the RGB channel
        squeezed = np.reshape(video_buffer, (self.video_buff_size * 3, self.image_size, self.image_size))
        if self.Display_loading_video == True:
            # x, y = 0, 10  # Position of the text
            # # Font settings
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 0.4
            # font_color = (255, 255, 255)  # White color
            # font_thickness = 1
            # cv2.putText(this_resize, this_label, (x, y), font, font_scale, font_color, font_thickness)
            cv2.imshow("First Frame R", squeezed[0, :, :].astype((np.uint8)))
            cv2.imshow("First Frame G", squeezed[1, :, :].astype((np.uint8)))
            cv2.imshow("First Frame B", squeezed[2, :, :].astype((np.uint8)))
            cv2.imshow("First Frame R1", squeezed[30, :, :].astype((np.uint8)))
            cv2.imshow("First Frame G1", squeezed[31, :, :].astype((np.uint8)))
            cv2.imshow("First Frame B1", squeezed[32, :, :].astype((np.uint8)))
            cv2.imshow("First Frame R2", squeezed[60, :, :].astype((np.uint8)))
            cv2.imshow("First Frame G2", squeezed[61, :, :].astype((np.uint8)))
            cv2.imshow("First Frame B2", squeezed[62, :, :].astype((np.uint8)))
            cv2.waitKey(1)
        return video_buffer, squeezed,flow_buffer,Valid_video
    def read_a_batch(self, this_epoch=0): # use this epoch to slide the loader window in data is longer than that
        if self.Read_from_pkl == False:
            folder_path = Dataset_video_root
            file_name_extention = ".mp4"
        else:
            if MICCAI_data == True:
                folder_path = Dataset_video_pkl_root
                if Evaluation_slots == True:
                    folder_path = Dataset_video_pkl_MICCAI_test

            if MICCAI_data_merge == True:
                folder_path = Dataset_video_pkl_merge_root
            if Cholec_data_flag == True:
                folder_path = Dataset_video_pkl_cholec
            if Thoracic_data_flag == True:
                folder_path = Dataset_video_pkl_thoracic
            if Test_on_cholec_seg8k ==True:
                folder_path =  Dataset_video_pkl_cholec8k
            if Endovis_data == True:
                folder_path = Dataset_video_pkl_endovis
            if DAVIS_data == True:
                folder_path = Dataset_video_pkl_Davis
            if YTVOS_data == True:
                folder_path = Dataset_video_pkl_YTVOS
            if YTOBJ_data == True:
                folder_path = Dataset_video_pkl_YTOBJ
            if MICCAIv2 == True:
                folder_path = Dataset_video_pkl_MICCAIv2

            file_name_extention = ".pkl"
        self.features=[]
        self.slots_prototype =[]
        self.slots_prototype_left =[]
        self.slots_prototype_right = []
        self.GT_slots = []
        self.GT_slots_ordered=[]
        self.slot_masks = []
        self.input_videos=[]
        self.labels = []
        self.this_label_mask = None
        self.all_raw_labels=[]
        self.this_label_string=[]
        self.this_file_name = None
        self.this_label = None
        i=0
        while(1): # load a batch of images
            self.features_exist = False
            start_time = time()

            index = self.read_record
            filename, dataset_source = self.all_video_dir_list[index]  # Retrieve filename and source
            print(dataset_source+filename)
            # Determine folder_path based on dataset source
            if dataset_source == 'cholec':
                folder_path = Dataset_video_pkl_cholec
            elif dataset_source == 'thoracic':
                folder_path = Dataset_video_pkl_thoracic
            elif dataset_source == 'miccai':
                folder_path = Dataset_video_pkl_root
            elif  dataset_source == 'endovis':
                folder_path = Dataset_video_pkl_endovis
            # Add other dataset sources as needed
            if filename.endswith(file_name_extention):
                # Extract clip ID from the filename
                self.this_file_name = filename
                # clip_id = int(filename.split("_")[1].split(".")[0])
                clip_name = filename.split('.')[0]
                # if clip_name=="clip_000189":
                #     a111=0
                #     a222=a111
                # clip_name = 'clip_001714'
                # filename =  'clip_001714.mp4'
                # label_Index = labels.index("clip_"+str(clip_id))
                # Check if the clip ID is within the range you want to read
                # if clip_id <= num_clips_to_read:
                # Construct the full path to the video clip
                video_path = os.path.join(folder_path, filename)
                if self.Read_from_pkl == False:
                    self.video_buff, self.video_buff_s,self.flow_buffer, Valid_video_flag = self.load_this_video_buffer(video_path)

                    if self.Save_pkl == True and Valid_video_flag == True:
                        this_video_buff = self.video_buff.astype((np.uint8))
                        this_flow_buff = self.flow_buffer.astype((np.uint8))

                        io.save_a_pkl(Dataset_video_pkl_root, clip_name, this_video_buff)
                        io.save_a_pkl(Dataset_video_pkl_flow_root, clip_name, this_flow_buff)
                        
                else:
                    # if clip_name!="clip_000189":
                    if dataset_source == 'miccai' or MICCAI_data_merge:
                        if Evaluation_slots == False:
                            if MICCAI_data_merge:
                                # Dataset_video_pkl_root = Dataset_video_pkl_merge_root
                                this_video_buff = io.read_a_pkl(Dataset_video_pkl_merge_root, clip_name)
                            else:
                                this_video_buff = io.read_a_pkl(Dataset_video_pkl_root, clip_name)

                            self.video_buff = this_video_buff 
                            if clip_name in self.all_labels:
                                this_label = self.all_labels[clip_name]
                                print(this_label)
                                self.this_label_string. append (this_label)
                                binary_vector = np.array([1 if category in this_label else 0 for category in categories], dtype=int)
                                # seperate the binary vector as left and right channel, so that when the image is fliped, two vector will exchange
                                binary_vector_l, binary_vector_r = self.convert_left_right_v(this_label)
                            Valid_video_flag = True
                        else:
                            data_dict = io.read_a_pkl(Dataset_video_pkl_MICCAI_test, clip_name)
                            this_video_buff = data_dict['frames']
                            
                            labels= data_dict['labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k
                            # labels= data_dict['super_labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k

                            self.this_frame_label = labels
                            
                            # self.video_buff = this_video_buff[:,0:self.cut_video_len,:,:]
                            self.video_buff = this_video_buff 


                    if MICCAIv2 == True:
                        data_dict = io.read_a_pkl(Dataset_video_pkl_MICCAIv2, clip_name)
                        this_video_buff = data_dict['frames']
                        
                        video_label= data_dict['tool_presence']      # (7,29) for 
                        self.video_buff = this_video_buff 
                        Valid_video_flag = True

                    if dataset_source == 'cholec':                     
                        data_dict = io.read_a_pkl(Dataset_video_pkl_cholec, clip_name)
                        this_video_buff = data_dict['frames']
                        
                        labels= data_dict['labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k
                        self.this_frame_label = labels
                        
                        self.video_buff = this_video_buff 
                        Valid_video_flag = True
                    if DAVIS_data == True:                     
                        data_dict = io.read_a_pkl(Dataset_video_pkl_Davis, clip_name)
                        this_video_buff = data_dict['frames']
                        
                        labels= data_dict['labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k
                        # labels= data_dict['super_labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k

                        self.this_frame_label = labels
                        
                        self.video_buff = this_video_buff 
                    if YTVOS_data == True:                     
                        data_dict = io.read_a_pkl(Dataset_video_pkl_YTVOS, clip_name)
                        this_video_buff = data_dict['frames']
                        
                        labels= data_dict['labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k
                        # labels= data_dict['super_labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k

                        self.this_frame_label = labels
                        
                        self.video_buff = this_video_buff
                    if YTOBJ_data== True:                     
                        data_dict = io.read_a_pkl(Dataset_video_pkl_YTOBJ, clip_name)
                        # this_video_buff = data_dict['masked_frames']
                        this_video_buff = data_dict['frames']

                        
                        labels= data_dict['labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k
                        # labels= data_dict['super_labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k

                        # self.this_frame_label = labels
                        self.this_frame_label = labels[0:self.video_len]
                        self.this_masks = data_dict['masks'][0:self.video_len]
                        self. this_boxs = data_dict['boxs'][0:self.video_len]
                        
                        self.video_buff = this_video_buff 
                    if dataset_source == 'thoracic':
                        data_dict = io.read_a_pkl(Dataset_video_pkl_thoracic, clip_name)
                        this_video_buff = data_dict['frames'] 
                        
                        labels= data_dict['labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k
                        # self.this_frame_label = labels
                        
                        self.video_buff = this_video_buff 
                        Valid_video_flag = True
                    if dataset_source == 'endovis':
                        data_dict = io.read_a_pkl(Dataset_video_pkl_endovis, clip_name)
                        this_video_buff = data_dict['frames']                        
                        labels= data_dict['labels']      # (7,29) for cholec , (13, 29, 256, 256) for seg8k
                        # self.this_frame_label = labels
                        self.video_buff = this_video_buff
                        Valid_video_flag = True

                    if self.Load_flow == True:
                        # if clip_name=="clip_000189":
                        #     pass
                        # else:

                            this_flow_buff = io.read_a_pkl(Dataset_video_pkl_flow_root, clip_name)
                            self.flow_buffer = this_flow_buff
                    if self.Load_feature == True:
                            # if Cholec_data_flag:
                            #     this_features = io.read_a_pkl(output_folder_sam_feature, clip_name)
                            # else:
                            this_features=[]
                            this_features = io.read_a_pkl(sam_feature_OLG_dir, clip_name)
                            if this_features is None:
                                self.features_exist = False
                            else:
                                self.this_features=this_features
                                this_features = this_features.permute(1,0,2,3).float()
                                this_features = this_features.to(self.device)
                                self.features_exist = True
                    if  Load_prototype == True:
                        Valid_video_flag = False
                        if (((clip_name+'.csv') in self.all_prototype_label_list) and Load_prototype_label) or Load_prototype_label==False: 

                            data_dict = io.read_a_pkl(Slots_prototype_pkl, clip_name)
                            # this_prototype = data_dict['prototype']
                            average_slots_list = []
                            average_slots_list_l = []
                            average_slots_list_r = []

                            slot_dim =128
                            if Load_prototype_label:
                                file_path = Slots_label_dir +(clip_name+'.csv')# Replace with the path to your CSV file
                                df = pd.read_csv(file_path)

                                # Extract relevant columns: 'slot' and 'tools_present'
                                # Drop any rows with NaN values and reset the index
                                relevant_data = df[['slot', 'tools_present']].dropna().reset_index(drop=True)

                                # Create lists for slots and tools_present
                                slots = relevant_data['slot'].tolist()
                                tools_present = relevant_data['tools_present'].tolist()
                                one_hot_vectors = relevant_data['tools_present'].apply(lambda x: convert_to_one_hot(x, categories)).tolist()
                            new_vectors = []
                            new_index  = 0
                            for cluster_label in data_dict['prototype']:
                                if np.isnan(data_dict['prototype'][cluster_label][2]).any():
                                    # Replace NaN with a zero vector of the appropriate dimension
                                    # average_slot = np.zeros(slot_dim*3)
                                    average_slot = np.zeros(slot_dim)

                                else:
                                    N = len ( data_dict['prototype'][cluster_label][0])  # Replace 10 with any positive integer
                                    random_number = random.randint(0, N-1)
                                    random_slot = data_dict['prototype'][cluster_label][0][random_number]
                                    # random_slot = data_dict['prototype'][cluster_label][0][0] # just pick the top 1

                                    avg_slot = data_dict['prototype'][cluster_label][3]
                                    # average_slot = np.concatenate((avg_slot,random_slot))
                                    average_slot =  random_slot


                                average_slots_list.append(average_slot)
                                if data_dict['prototype'][cluster_label][4] == False:
                                    average_slots_list_l.append(average_slot)
                                else:
                                    average_slots_list_r.append(average_slot)
                                if Load_prototype_label:

                                    new_vectors. append (one_hot_vectors[cluster_label])
                                 

                            this_GT_slots = data_dict['GT_slot_BG']
                            # Stack the average slots into a single array
                            stacked_average_slots = np.stack(average_slots_list, axis=0)
                            stacked_average_slots_l = np.stack(average_slots_list_l, axis=0)
                            stacked_average_slots_r = np.stack(average_slots_list_r, axis=0)

                            self.slots_prototype.append (stacked_average_slots)
                            self.slots_prototype_left.append (stacked_average_slots_l)
                            self.slots_prototype_right.append (stacked_average_slots_r)

                            self.GT_slots.append (this_GT_slots)
                            self.slot_masks. append (data_dict['prototype'] )
                            if Load_prototype_label:
                                
                                stacked_one_hot_vectors = np.stack(new_vectors, axis=0)
                                self.GT_slots_ordered.append(stacked_one_hot_vectors)
                        # Loop through each cluster label and collect the average slots
                            Valid_video_flag = True
                # clip_name= 'test'
        

                if ( (dataset_source == 'miccai' and (clip_name in self.all_labels)) or Valid_video_flag==True):  
                    if self.Load_feature == True:
                        self.features.append(this_features)
                    if dataset_source == 'miccai' and Evaluation_slots == True:
                        mask,frame_label,video_label = format_convertor.label_from_Miccaitest(labels)
                        # mask,frame_label,video_label = format_convertor.label_from_thoracic2cholec(labels)

                        binary_vector = video_label
                        self.this_video_label = binary_vector
                        self.this_frame_label = frame_label
                        self.this_label_mask = mask 
                        self.this_raw_labels = frame_label
                        binary_vector_l = 0
                        binary_vector_r = 0

                        pass
                    
                    # if MICCAI_data == True:
                    if MICCAIv2 == True:
                        binary_vector = video_label
                         
                        binary_vector_l = 0
                        binary_vector_r = 0
                    if dataset_source == 'cholec':
                        binary_vector = self.merge_labels(labels)
                        if Test_on_cholec_seg8k:
                            mask,frame_label,video_label = format_convertor.label_from_seg8k_2_cholec(labels)
                            binary_vector = video_label
                            self.this_video_label = binary_vector
                            self.this_frame_label = frame_label
                            self.this_label_mask = mask 
                            self.this_raw_labels = frame_label
                        binary_vector_l = 0
                        binary_vector_r = 0
                    if DAVIS_data == True:
                        binary_vector = self.merge_labels(labels)
                         
                        binary_vector_l = 0
                        binary_vector_r = 0
                    if YTVOS_data == True:
                        binary_vector = self.merge_labels(labels)
                         
                        binary_vector_l = 0
                        binary_vector_r = 0
                    if YTOBJ_data == True:
                        binary_vector = self.merge_labels(labels)
                        self.this_video_label = binary_vector
                        
                        binary_vector_l = 0
                        binary_vector_r = 0
                    if dataset_source == 'thoracic' :
                        if  Evaluation_slots == True: 
                            mask,frame_label,video_label = format_convertor.label_from_thoracic(labels)
                        else:
                            mask=None
                            frame_label =0
                            video_label= np.ones(len(categories))
                        binary_vector = video_label
                        self.this_video_label = binary_vector
                        self.this_frame_label = frame_label
                        self.this_label_mask = mask 
                        self.this_raw_labels = frame_label
                        binary_vector_l = 0
                        binary_vector_r = 0
                    if dataset_source == 'endovis':
                        mask,frame_label,video_label = format_convertor.label_from_endovis(labels)
                        # mask,frame_label,video_label = format_convertor.label_from_thoracic2cholec(labels)

                        binary_vector = video_label
                        self.this_video_label = binary_vector
                        self.this_frame_label = frame_label
                        self.this_label_mask = mask
                        self.this_raw_labels = frame_label
                        binary_vector_l = 0
                        binary_vector_r = 0

                        pass


                    # load the squess and unsquess
                    self.this_label = binary_vector
                    if dataset_source == 'cholec' or dataset_source == 'thoracic' or dataset_source == 'endovis' or DAVIS_data==True:
                        self.this_raw_labels = self.this_frame_label
                        self.all_raw_labels. append(self.this_frame_label)
                    if self.Display_loading_video == True:
                        cv2.imshow("SS First Frame R", this_video_buff[0,15, :, :].astype((np.uint8)))
                        cv2.imshow("SS First Frame G", this_video_buff[1,15, :, :].astype((np.uint8)))
                        if self.Load_flow == True:
                            cv2.imshow("SS First Frame flow", this_flow_buff[15,:, :].astype((np.uint8)))
                        cv2.waitKey(1)

                    # fill the batch
                    # if Valid_video_flag == True:
                    # self.video_buff = basic_operator.random_verse_the_video(self.video_buff)
                    # self.motion = basic_operator.compute_optical_flow(self.video_buff)
                    if Data_aug == True:
                        flag =  random.choice([True, False])
                        flip_flag = random.choice([True, False])
                        
                    else:
                        flag = False
                        flip_flag = False
                    Crop_flag = random.choice([True,True, False])
                    if Crop_half==True  and Crop_flag==True:
                        self.video_buff=basic_operator.half_crop(self.video_buff) # for slot attention
                        # if      binary_vector [9]==1: #
                        #     self.video_buff=basic_operator.half_crop(self.video_buff)
                        # else:
                        #     self.video_buff=basic_operator.random_mask_or_crop(self.video_buff)
                    if flag ==True:
                        
                        self.video_buff,used_angle=basic_operator.random_augment(self.video_buff)
                        if self.Load_flow==True:
                            self.flow_buffer = basic_operator.rotate_buff(self.flow_buffer,angle=used_angle )
                    if Random_mask==True:
                        self.video_buff=basic_operator.hide_patch(self.video_buff)
                    if Random_Full_mask == True:
                        self.video_buff=basic_operator.hide_full_image(self.video_buff)


                    # self.video_buff[0,:,:,:]= self.motion 
                    # self.video_buff[1,:,:,:]= self.motion 
                    # self.video_buff[2,:,:,:]= self.motion 
                    if Mask_out_partial_label == True:
                        # binary_vector[0] =0

                        # binary_vector[1] =0
                        # binary_vector[2] = 0
                        # binary_vector[3] = 0
                        # binary_vector[4] = 0
                        # binary_vector[5] =0
                        # binary_vector[6] =0
                        binary_vector = binary_vector*label_mask
                        # binary_vector[11] =0
                        # binary_vector[12] =0

                    if (Select_on_label == True and np.all(binary_vector[removed_category] == 0)) or Select_on_label == False:
                        # flip_flag = True
                        if flip_flag == False:
                            # self.input_videos[i,:, :, :, :] = self.video_buff
                            if self.Load_flow == True:
                                self.input_flows[i, :, :, :] = self.flow_buffer
                            self.labels.append ( binary_vector)
                            if dataset_source == 'miccai' and Evaluation_slots==False:
                                self.labels_LR[i, :] = np.concatenate([binary_vector_l, binary_vector_r])
                        # self.labels_LR[i, 1, :] = binary_vector_r
                        else:
                            self.video_buff= np.flip(self.video_buff, axis=3)
                            if self.Load_flow == True:
                                self.input_flows[i, :, :, :] =  np.flip(self.flow_buffer,axis=2)
                            self.labels.append ( binary_vector)
                            if dataset_source == 'cholec':
                                pass
                                # self.labels_LR[i, :] = np.concatenate([binary_vector_r, binary_vector_l])

                        # change the starting and ending point for cutting dataset:
                        # Video_down_sample_f = 15
                        # Check if evaluation is required and adjust cut points
                        if Evaluation == True or Evaluation_slots == True:
                            cut_start = 0
                            cut_end = self.cut_video_len   # Scale cut_end with downsample factor
                        else:
                            video_len_og = int(self.video_buff.shape[1] / Video_down_sample_f)
                            shift = 0
                            if video_len_og > self.cut_video_len :
                                # Scale the shift with downsample factor
                                shift = int(this_epoch % (video_len_og - self.cut_video_len  ))
                                shift = np.clip(shift, 0, video_len_og - self.cut_video_len )

                            # Apply the scaled cut_start and cut_end
                            cut_start = 0 + shift
                            cut_end = self.cut_video_len+ shift

                        # Downsample the video buffer if needed
                        downsampled_video_buff = self.video_buff[:, ::Video_down_sample_f, :, :]  # Downsample along the time axis (second axis)

                        # Apply the cut on the downsampled video buffer
                        if self.this_label_mask is not None:
                            # If masks exist, downsample them as well and apply the cut
                            self.this_label_mask = self.this_label_mask[:, ::Video_down_sample_f, :, :]
                            self.this_label_mask = self.this_label_mask[:, cut_start:cut_end, :, :]  # Cut here

                        # Append the downsampled and cut video frames to the input videos list
                        self.input_videos.append(downsampled_video_buff[:, cut_start:cut_end, :, :])  # Cut 
                        i+=1
                    else:
                        print ("label is removed")
                    

                else:
                    print("Key does not exist in the dictionary.")

            end_time = time()


            print(self.read_record)
            # print("time is :" + str(end_time - start_time))
            self.read_record +=1
            if self.read_record>= self.video_num:
                print("all videos have been readed")
                # random.shuffle(self.all_video_dir_list)
                 
                self.all_read_flag = 1
                self.read_record =0
                random.shuffle(self.all_video_dir_list)
             
            if (i>=self.batch_size):
                        break
        if self.Load_feature == True:
            if self.features_exist:
                self.features = torch.stack(self.features, dim=0)
            else:
                self.features = None
        if  Load_prototype:
            # self.slots_prototype.append (stacked_average_slots)
            self.slots_prototype = np.stack(self.slots_prototype, axis=0) 
            self.GT_slots = np.stack(self.GT_slots, axis=0) 
            if Load_prototype_label:
                self.GT_slots_ordered =  np.stack(self.GT_slots_ordered, axis=0) 
        self.input_videos = self.Align_stack_videos(self.input_videos )
        self.labels =  np.stack(self.labels, axis=0) 
        # return self.input_image,self.input_path# if out this folder boundary, just returen
        this_pointer = 0
        # i = self.read_record
        # this_folder_list = self.folder_list[self.folder_pointer]
        # # read_end  = self.read_record+ self.batch_size
        # this_signal = self.signal[self.folder_pointer]
        if Seperate_LR == False:
            return self.input_videos, self.labels
        else:
            return self.input_videos, self.labels_LR

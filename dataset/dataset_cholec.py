#THe data set read the PKL file for Contour with sheath so the contact can be detected
import cv2
import numpy as np
import csv
import re
import os
print("Current working directory:", os.getcwd())
from time import  time
import dataset.io as io
import random
import image_operator.basic_operator as basic_operator
 
from working_dir_root import Dataset_video_root, Dataset_label_root, Dataset_video_pkl_root,Dataset_video_pkl_flow_root,Batch_size,Random_mask
Dataset_video_root =  "C:/2data/cholec80/frames/"
Dataset_label_root =  "C:/2data/cholec80/tool_annotations/"
Batch_size = 1
Seperate_LR = False
Mask_out_partial_label = True
input_ch = 3 # input channel of each image/video


categories_count = [17, 13163, 17440, 576, 1698, 4413, 11924, 10142, 866, 2992,131, 17, 181, 1026]

total_samples = sum(categories_count)
class_weights = [total_samples / (abs(count) * len(categories_count)) for count in categories_count]

# weight_tensor = torch.tensor(class_weights, dtype=torch.float)

categories = [
    'Grasper', #0   - 17
    'Bipolar', #1     -13163
    'Hook', #2     -17440
    'Scissors', #3        -576
    'Clipper',#4         - 1698
    'Irrigator',#5     -4413
    'SpecimenBag',#6     -11924               
]
Obj_num = len(categories)
class myDataloader_cholec(object):
    def __init__(self, OLG=False,img_size = 128,Display_loading_video = False,
                 Read_from_pkl= True,Save_pkl = False,Load_flow =False):
        print("GPU function is : "+ str(cv2.cuda.getCudaEnabledDeviceCount()))
        self.image_size = img_size
        self.Display_loading_video =Display_loading_video
        self.Read_from_pkl= Read_from_pkl 
        self.Save_pkl=Save_pkl
        self.batch_size = Batch_size
        self.obj_num = Obj_num
        self.Load_flow=Load_flow
        self.video_down_sample = 60  # 60 FPS
        self.video_len = 29
        self.video_buff_size = int(60/self.video_down_sample) * self.video_len  # each video has 30s discard last one for flow
        self.OLG_flag = OLG
        self.GT = True
        self.noisyflag = False
        self.Random_rotate = True
        self.Random_vertical_shift = True
        self.input_images= np.zeros((self.batch_size, 1, img_size, img_size))
        self.input_videos = np.zeros((self.batch_size,3,self.video_buff_size,img_size,img_size )) # RGB together
        self.input_flows = np.zeros((self.batch_size,self.video_buff_size,img_size,img_size )) # RGB together

        # the number of the contour has been increased, and another vector has beeen added
        self.labels_LR= np.zeros((self.batch_size,2*self.obj_num))  # predifine the path number is 2 to seperate Left and right
        self.labels= np.zeros((self.batch_size, self.obj_num))  # left right merge

        self.all_read_flag =0
        self.save_id =0
        self.read_record = 0
        self.all_labels = self.load_all_lables()
        if Read_from_pkl == False:
            self.all_video_dir_list = os.listdir(Dataset_video_root)
        else:
            self.all_video_dir_list = os.listdir(Dataset_video_pkl_root)

        self.video_num = len (self.all_video_dir_list)

        #Guiqiu modified for my computer
        # self.com_dir =  Generator_Contour_sheath().com_dir # this dir is for the OLG
        # if self.OLG_flag == True:
        #      # initial lizt the
        #     self.talker = Communicate()
    def load_all_lables(self): # load all labels and save then as dictionary format
        csv_file_path = Dataset_label_root + "labels.csv"

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

    # load one video buffer (self.video_buff_size , 3, img_size, img_size),
    # and its squeesed which RGB are put together (self.video_buff_size * 3, img_size, img_size),
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
    def read_a_batch(self):
        if self.Read_from_pkl == False:
            folder_path = Dataset_video_root
            file_name_extention = ".mp4"
        else:
            folder_path = Dataset_video_pkl_root
            file_name_extention = ".pkl"

        for i in range(self.batch_size): # load a batch of images
            start_time = time()

            index = self.read_record
            filename = self.all_video_dir_list[index]
            print(filename)

            if filename.endswith(file_name_extention):
                # Extract clip ID from the filename
                clip_id = int(filename.split("_")[1].split(".")[0])
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
                    this_video_buff = io.read_a_pkl(Dataset_video_pkl_root, clip_name)
                    self.video_buff = this_video_buff[:,0:self.video_len,:,:]
                    if self.Load_flow == True:
                        # if clip_name=="clip_000189":
                        #     pass
                        # else:

                            this_flow_buff = io.read_a_pkl(Dataset_video_pkl_flow_root, clip_name)
                            self.flow_buffer = this_flow_buff


                    Valid_video_flag = True
                # clip_name= 'test'

                if clip_name in self.all_labels and Valid_video_flag==True:
                    this_label = self.all_labels[clip_name]
                    print(this_label)

                    binary_vector = np.array([1 if category in this_label else 0 for category in categories], dtype=int)
                    # seperate the binary vector as left and right channel, so that when the image is fliped, two vector will exchange
                    binary_vector_l, binary_vector_r = self.convert_left_right_v(this_label)
                    # load the squess and unsquess

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
                    flag =  random.choice([True, False])
                    if flag ==True:
                        
                        self.video_buff,used_angle=basic_operator.random_augment(self.video_buff)
                        if self.Load_flow==True:
                            self.flow_buffer = basic_operator.rotate_buff(self.flow_buffer,angle=used_angle )
                    if Random_mask==True:
                        self.video_buff=basic_operator.hide_patch(self.video_buff)

                    flip_flag = random.choice([True, False])
                    # self.video_buff[0,:,:,:]= self.motion 
                    # self.video_buff[1,:,:,:]= self.motion 
                    # self.video_buff[2,:,:,:]= self.motion 
                    if Mask_out_partial_label == True:
                        binary_vector[0] =0
                        binary_vector[3] = 0
                        binary_vector[4] = 0
                        binary_vector[5] = 0
                        binary_vector[8] =0
                        binary_vector[10] =0
                        binary_vector[11] =0
                        binary_vector[12] =0


                    # flip_flag = True
                    if flip_flag == False:
                        self.input_videos[i,:, :, :, :] = self.video_buff
                        self.input_flows[i, :, :, :] = self.flow_buffer
                        self.labels[i, :] = binary_vector
                        self.labels_LR[i, :] = np.concatenate([binary_vector_l, binary_vector_r])
                    # self.labels_LR[i, 1, :] = binary_vector_r
                    else:
                        self.input_videos[i, :, :, :, :] = np.flip(self.video_buff, axis=3)
                        self.input_flows[i, :, :, :] =  np.flip(self.flow_buffer,axis=2)

                        self.labels[i, :] = binary_vector
                        self.labels_LR[i, :] = np.concatenate([binary_vector_r, binary_vector_l])

                else:
                    print("Key does not exist in the dictionary.")

            end_time = time()


            print(self.read_record)
            # print("time is :" + str(end_time - start_time))
            self.read_record +=1
            if self.read_record>= self.video_num:
                print("all videos have been readed")
                self.all_read_flag = 1
                self.read_record =0

            pass

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

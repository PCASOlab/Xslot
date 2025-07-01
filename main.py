# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
from time import time
import os
 

# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_endovis'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_miccai'  # Change this to your target mode
#
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_cholec'  # Change this to your target mode
os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_thoracic'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_thoracic'  # Change this to your target mode



# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_cholec'  # Change this to your target mode

print("Current working directory:", os.getcwd())
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import eval_slots
from model import    model_infer_slot_att
from working_dir_root import Output_root,Linux_computer
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
from working_dir_root import Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
from working_dir_root import Enable_student,Batch_size,selected_data,Display_down_sample, Data_percentage,Gpu_selection,Evaluation_slots,Max_epoch
from dataset import io
import pathlib
import argparse

# GPU_mode= True
# Continue_flag = True
# Visdom_flag = False
# Display_flag = False
# loadmodel_index = '3.pth'
dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
Output_root = Output_root+ "0Merge_predict" + dataset_tag + str(Data_percentage) + "/"
io.self_check_path_create(Output_root)
RESULT_FINISHED = 0
RESULT_TIMEOUT = 1

CHECKPOINT_SUBDIR = "checkpoints"
TENSORBOARD_SUBDIR = "tb"
METRICS_SUBDIR = "metrics"

# Define default configuration values
# DEFAULT_CONFIG = "video_SA/configs/videosaur/ytvis3_MLP_dinov2.yml"
# DEFAULT_CONFIG = "video_SA/configs/videosaur/ytvis3_MLP.yml"
DEFAULT_CONFIG = "model_config.yml"

# DEFAULT_CONFIG = "video_SA/configs/videosaur/ytvis3_MLP_cholec.yml"

# DEFAULT_CONFIG = "video_SA/configs/videosaur/ytvis3.yml"


DEFAULT_LOG_DIR = "./logs"

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
group.add_argument("-q", "--quiet", action="store_true", help="Suppress outputs")
parser.add_argument("-n", "--dry", action="store_true", help="Dry run (no logfiles)")
parser.add_argument(
    "--no-interactive", action="store_true", help="If running in non-interactive environment"
)
parser.add_argument("--no-tensorboard", action="store_true", help="Do not write tensorboard logs")
parser.add_argument(
    "--check-validation", action="store_true", help="Run correctness checks on data used during eval"
)
parser.add_argument(
    "--run-eval-after-training", action="store_true", help="Evaluate after training has stopped"
)
parser.add_argument(
    "--use-optimizations", action="store_true", help="Enable Pytorch performance optimizations"
)
parser.add_argument("--timeout", help="Stop training after this time (format: DD:HH:MM:SS)")
parser.add_argument("--data-dir", help="Path to data directory")
parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Path to log directory")
parser.add_argument(
    "--no-sub-logdirs", action="store_true", help="Directly use log dir to store logs"
)
parser.add_argument(
    "--continue",
    dest="continue_from",
    type=pathlib.Path,
    help="Continue training from this log folder or checkpoint path",
)
parser.add_argument("--config_overrides_file", help="Configuration to override")
parser.add_argument(
    "config", nargs="?", default=DEFAULT_CONFIG, help="Configuration to run"
)
parser.add_argument("config_overrides", nargs="*", help="Additional arguments")

import pickle

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
   
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
if GPU_mode ==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")

# dataroot = "../dataset/CostMatrix/"
# torch.set_num_threads(8)
 # create the model

if Visdom_flag == True:
    from visual import VisdomLinePlotter

    plotter = VisdomLinePlotter(env_name='path finding training Plots')
 
 
   
Model_infer = model_infer_slot_att._Model_infer(parser.parse_args(),GPU_mode,num_gpus,Using_contrast=False,Using_SP_regu = False,Using_SP = True,Using_slot_bert=True,slot_ini= "binder+merger",Sim_threshold=0.90,gpu_selection=Gpu_selection,pooling="max",TPC=True)
device = Model_infer.device
 
dataLoader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='else',Device=device)

if Continue_flag == True:
    Model_infer.model.load_state_dict(torch.load(Output_root + 'model' + loadmodel_index ))
  



 
read_id = 0
 

epoch = 0
 
iteration_num = 0
#################
#############training
saver_id =0
displayer = Display(parser.parse_args())
epoch =0
features =None
visdom_id=0
while (1):
    start_time = time()
    input_videos, labels= dataLoader.read_a_batch(this_epoch= epoch)
    input_videos_GPU = torch.from_numpy(np.float32(input_videos))
    labels_GPU = torch.from_numpy(np.float32(labels))
    input_videos_GPU = input_videos_GPU.to (device)
    labels_GPU = labels_GPU.to (device)
    input_flows = dataLoader.input_flows*1.0/ 255.0
    input_flows_GPU = torch.from_numpy(np.float32(input_flows))  
    input_flows_GPU = input_flows_GPU.to (device)
    if Load_feature ==True:
        features = dataLoader.features.to (device)
    Model_infer.forward(input_videos_GPU,input_flows_GPU,features,Enable_student,epoch=epoch)
    # Model_infer.forward(input_videos_GPU,input_flows_GPU,features,Enable_student)

 
    if Evaluation == False and Evaluation_slots==False:
        Model_infer.optimization(labels_GPU,Enable_student) 

    if  Save_feature_OLG== True:
        this_features= Model_infer.f[Batch_size-1].permute(1,0,2,3).half()
        sam_pkl_file_name = dataLoader.this_file_name
        sam_pkl_file_path = os.path.join(sam_feature_OLG_dir, sam_pkl_file_name)

        with open(sam_pkl_file_path, 'wb') as file:
            pickle.dump(this_features, file)
            print("sam Pkl file created:" +sam_pkl_file_name)
    if Save_sam_mask == True:
         
        this_mask= Model_infer.sam_mask.half()
        mask_pkl_file_name = dataLoader.this_file_name
        mask_pkl_file_path = os.path.join(output_folder_sam_masks, mask_pkl_file_name)

        with open(mask_pkl_file_path, 'wb') as file:
            pickle.dump(this_mask, file)
            print("sam Pkl file created:" +mask_pkl_file_name)


    if Display_flag == True and read_id%Display_down_sample == 0:
        dataLoader.labels  = dataLoader.labels * 0 +1
        displayer.train_display(Model_infer,dataLoader,read_id,Output_root)
        print(" epoch" + str (epoch) )
        

    
        

        # break
    

    if Evaluation == False and Evaluation_slots == False:
        
        if read_id % 50== 0 and Visdom_flag == True  :
            
            plotter.plot('l0', 'l0', 'l0', visdom_id, Model_infer.lossDisplay.cpu().detach().numpy())
            # if Enable_student:
            plotter.plot('1ls', '1ls', 'l1s', visdom_id, Model_infer.lossDisplay_s.cpu().detach().numpy())
        if read_id % 1== 0   :
            print(" epoch" + str (epoch) )
            print(" loss" + str (Model_infer.lossDisplay.cpu().detach().numpy()) )
            if Enable_student:
                print(" loss_SS" + str (Model_infer.lossDisplay_s.cpu().detach().numpy()) )

    if (read_id % 1000) == 0  :
        torch.save(Model_infer.model.state_dict(), Output_root + "model" + str(saver_id) + ".pth")
         

        saver_id +=1
        if saver_id >1:
            saver_id =0

        end_time = time()

        print("time is :" + str(end_time - start_time))

    read_id+=1
    visdom_id+=1
    if dataLoader.all_read_flag ==1:
        Save_feature_OLG = False
        #remove this for none converting mode
        epoch +=1

        print("finished epoch" + str (epoch) )
        dataLoader.all_read_flag = 0
        read_id=0

        if Evaluation:
            break
        if Save_feature_OLG: 
            break
        if epoch > Max_epoch  :
            output_file = eval_slots.process_metrics_from_excel(Output_root + "/metrics_video.xlsx", Output_root)

            break
     




















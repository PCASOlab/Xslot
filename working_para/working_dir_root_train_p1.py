
# PC
# working_root = "/media/guiqiu/Installation/database/surgtoolloc2022_dataset/"
#
# Dataset_video_root =  working_root + "_release/training_data/video_clips/"
# Dataset_video_pkl_root = working_root + "_release/training_data/video_clips_pkl/"
# Dataset_label_root =  working_root + "_release/training_data/"
# config_root =   working_root + "config/"
# Output_root =   "/media/guiqiu/Installation/database/output/"
# #
import json
import os
# Remote
Linux_computer= True
working_root = "C:/2data/"
 
if Linux_computer == True:
    working_root = "/home/guiqiu/GQ_project/weakly_supervised/Data/"

working_pcaso_raid = "/media/guiqiu/Weakly_supervised_data/"


Dataset_video_root =  working_root + "training_data/video_clips/"

Dataset_video_pkl_merge_root  = working_root  + "training_data/video_clips_pkl_merge/"
Dataset_video_pkl_flow_root = working_root + "training_data/video_clips_pkl_flow/"
Dataset_video_pkl_cholec = working_root + "training_data/video_clips_pkl_cholec/"
Dataset_video_pkl_cholec = working_root + "cholec80/output_pkl/"
Dataset_video_pkl_thoracic =  'C:/2data/Raw_data_Chrocic/data/' + 'output_pkl/'
Dataset_video_pkl_endovis = "C:/2data/endovis/2017/pkl/"


if Linux_computer == True:
    # Dataset_video_pkl_cholec = working_pcaso_raid + "cholec80/output_pkl/"
    Dataset_video_pkl_cholec = working_pcaso_raid + "cholec80/output_pkl_croped/"


Dataset_label_root =  working_root + "training_data/"
config_root =   working_root + "config/"
Output_root =   working_pcaso_raid+"output/"
SAM_pretrain_root = working_pcaso_raid+"output/SAM/"
DINO_pretrain_root = working_root + "output/DINO/"

output_folder_sam_feature = working_root+ "cholec80/output_sam_features/"
# Dataset_video_pkl_cholec8k =  working_root+ "cholecseg8k_working/output_pkl/"
Dataset_video_pkl_cholec8k =  working_root+ "cholecseg8k_working/output_pkl_croped/"


output_folder_sam_feature_cholec8k = working_root + "cholecseg8k_working/output_sam_features/"
output_folder_sam_masks = Output_root + "sam_masks"

train_test_list_dir = working_root + "output/train_test_list/"
train_sam_feature_dir = working_root+ "cholec80/train_sam_feature/"
sam_feature_OLG_dir= working_root+ "cholec80/sam_feature_OLG/"





if Linux_computer == True:
    output_folder_sam_feature = working_pcaso_raid+ "cholec80/output_sam_features/"
    # Dataset_video_pkl_cholec8k =  working_pcaso_raid+ "cholecseg8k_working/output_pkl/"
    Dataset_video_pkl_cholec8k =  working_pcaso_raid+ "cholecseg8k_working/output_pkl_croped/"

    Dataset_video_pkl_root = working_pcaso_raid + "MICCAI/video_clips_pkl/"
    Dataset_label_root =  working_pcaso_raid + "MICCAI/"
    
    Dataset_video_pkl_MICCAIv2 =  "/media/guiqiu/surgvu24pkl/"
    Dataset_video_pkl_endovis = working_pcaso_raid + "Endovis/pkl/"
    Dataset_video_pkl_MICCAI_test = working_pcaso_raid + "MICCAI_selected_GT/pkl/"
    # Dataset_video_pkl_thoracic = working_pcaso_raid + "choracic/pkl/"
    Dataset_video_pkl_thoracic = working_pcaso_raid + "Thoracic/pkl/"
    Thoracic_select =  working_pcaso_raid + "Thoracic/selected/"


    Davis_root = working_pcaso_raid + "DAVIS/"
    root_YTVOS = working_pcaso_raid + "YTVOS/"
    root_YTOBJ = working_pcaso_raid + "YTOBJ/"

    # root_YTOBJ = "C:/2data/TCAMdata/youtube video/download2/"
    Dataset_video_pkl_Davis = working_pcaso_raid + "DAVIS/pkl/"
    Dataset_video_pkl_YTVOS = working_pcaso_raid + "YTVOS/pkl/"

    Dataset_video_pkl_YTOBJ = working_pcaso_raid + "YTOBJ/pkl6/"

    Slots_prototype_pkl  =  working_pcaso_raid +  "out/pkl/"
    Slots_label_dir  =  working_pcaso_raid +  "out/labels/"



    output_folder_sam_feature_cholec8k = working_pcaso_raid + "cholecseg8k_working/output_sam_features/"
    train_sam_feature_dir = working_pcaso_raid+ "cholec80/train_sam_feature/"
    # sam_feature_OLG_dir= working_pcaso_raid+ "cholec80/sam_feature_OLG/"
    sam_feature_OLG_dir= working_pcaso_raid+ "sam_feature_OLG/"
    sam_feature_OLG_dir2= working_pcaso_raid+ "sam_feature_OLG2/"
    sam_feature_OLG_dir3= working_pcaso_raid+ "sam_feature_OLG3/"
    # sam_feature_OLG_dir = sam_feature_OLG_dir2


    video_saur_pretrain = working_pcaso_raid + "model_trained/videosaur-ytvis.ckpt"
    # video_saur_pretrain = working_pcaso_raid + "model_trained/videosaur_dinov2.ckpt"


    slot_out_dir = working_pcaso_raid + "MICCAI_slots_merged_mask"


data_flag_list = ["Cholec_data_flag", "Thoracic_data_flag", "MICCAI_data", "MICCAI_data_merge","MICCAIv2",
                  "Endovis_data","DAVIS","YTVOS","YTOBJ"]
selected_data= "Thoracic_data_flag"
if selected_data == "YTVOS":
    sam_feature_OLG_dir = os.path.join(root_YTVOS, "feature_OLG/")

if selected_data == "YTOBJ":
    sam_feature_OLG_dir = os.path.join(root_YTOBJ, "feature_OLG6/")
if selected_data == "Cholec_data_flag":
    sam_feature_OLG_dir= working_pcaso_raid+ "sam_feature_OLG2/"
if selected_data == "MICCAI_data_merge":
    sam_feature_OLG_dir= working_pcaso_raid+ "sam_feature_OLG_merge/"
Gpu_selection ='1'
Max_epoch = 200
Fintune= True  #

Evaluation = False
Evaluation_slots = False

img_size = 128
GPU_mode= True

Continue_flag = True
Test_on_cholec_seg8k= False

Visdom_flag = True
if Evaluation == True:
    Continue_flag = True

# 1.0, 0.9, 0.5, 0.3, 0.1 , 0.05, 0.01
#2.0 for fine tune
Data_percentage = 1.0
 
Display_flag = True
Display_down_sample = 10000
Display_student =False
Display_fuse_TC_ST = False
Display_final_SAM = False
Display_images= False
Display_visdom_figure=True
Display_embedding = False
Save_flag =False
loadmodel_index = '1.pth'

Random_mask_temporal_feature = False
Random_mask_patch_feature = False

Batch_size =1
Video_len = 5
Video_down_sample_f = 1
Data_aug = True
Crop_half = False
Random_mask = False
Random_Full_mask = False
Load_prototype = False
Load_prototype_label = False
Load_feature = False  # remember switch feature path
Save_feature_OLG = False

 
if Save_feature_OLG == True:
    Batch_size=1

Enable_student = False
 
if Evaluation:
    Enable_student = True
Save_sam_mask = False
Load_flow = False
Weight_decay =0.00001
Max_lr = 0.0001
learningR = 0.0001
min_lr =0.00001
learningR_res = 0.00001
Call_gradcam = False 
Use_max_error_rejection = False
Categories_json = os.path.join(Davis_root, 'Usupervised/DAVIS/categories.json')

# Try to open the file and load the categories, return None if the file is not found
def load_categories(Categories_json):
    try:
        with open(Categories_json, 'r') as f:
            category_list_json = json.load(f)
        Davis_categories_list = {key: value['id'] for key, value in category_list_json.items()}
        super_categories = {key: value['super_category'] for key, value in category_list_json.items()}
        Davis_super_category_list = list(set(super_categories.values()))
        Davis_super_category_list.sort()

        print("Categories:", Davis_categories_list)
        print("Super Categories:", Davis_super_category_list)
        return Davis_categories_list, Davis_super_category_list
    except FileNotFoundError:
        print(f"File not found: {Categories_json}")
        return None, None
# Check if the YTOBJ directory exists before trying to list its contents
def load_YTOBJ_categories(root_YTOBJ):
    directory = os.path.join(root_YTOBJ, "GroundTruth/")
    if os.path.exists(directory):
        return sorted(os.listdir(directory))
    else:
        print(f"Directory not found: {directory}")
        return None
# Call the function to load categories
Davis_categories_list, Davis_super_category_list = load_categories(Categories_json)

# Function to get all unique categories from a given JSON data
def get_all_unique_categories(json_data):
    videos = json_data.get('videos', {})
    all_categories = set()
    
    for video_data in videos.values():
        for obj_data in video_data.get('objects', {}).values():
            all_categories.add(obj_data['category'])
    
    return sorted(list(all_categories))

# Load the YTVOS category map
Json_map_dir = os.path.join(root_YTVOS, 'meta.json')

# Try to open the meta.json file and load the category map, return None if the file is not found
def load_category_map(Json_map_dir):
    try:
        with open(Json_map_dir, 'r') as f:
            category_map = json.load(f)
        return category_map
    except FileNotFoundError:
        print(f"File not found: {Json_map_dir}")
        return None

# Load YTVOS categories
category_map = load_category_map(Json_map_dir)
if category_map:
    all_categories = get_all_unique_categories(category_map)
    YTVOS_categories_list = all_categories
else:
    YTVOS_categories_list = None

# Load YTOBJ categories
YTOBJ_categories_list = load_YTOBJ_categories(root_YTOBJ)

# If the list is None, handle accordingly (e.g., skip further processing)
if YTOBJ_categories_list is None:
    print("No categories found, skipping further processing.")
else:
    # Continue with further processing
    print("YTOBJ Categories:", YTOBJ_categories_list)

# YTOBJ_categories_list = [
#     "aeroplane",
#     "bird",
#     "boat",
#     "car",
#     "cat",
#     "cow",
#     "dog",
#     "horse",
#     "motorbike",
#     "train"
# ]


class Para(object):
    def __init__(self):
        
        self.x=0

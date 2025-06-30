import cv2
import numpy as np
import json
import argparse
import os
from PIL import Image, ImageDraw
from files_io import save_a_image, save_a_pkl
import random
import pickle

video_format = ".mp4"

def generate_unique_colors(num_instances):
    """Generate unique colors for each instance."""
    random.seed(42)  # Ensure reproducibility
    colors = []
    for _ in range(num_instances):
        colors.append(tuple(random.randint(0, 255) for _ in range(3)))
    return colors
def load_parameters(task):
    with open('./src/data/parameters.json', 'r') as file:
        data = json.load(file)
    return data[task]
def generate_one_hot_instance_mask(num_instances, H, W, instance_masks):
    """
    Generate a one-hot mask for up to 7 instances.
    
    Args:
    - num_instances: Number of instances in the frame (maximum 7).
    - H: Height of the frame.
    - W: Width of the frame.
    - instance_masks: List of binary masks for each instance.

    Returns:
    - one_hot_mask: A one-hot encoded mask of shape (7, H, W), where each
                    channel corresponds to an instance mask.
    """
    max_instances = 7
    one_hot_mask = np.zeros((max_instances, H, W), dtype=np.uint8)

    # Assign each instance to a separate channel
    for i in range(min(num_instances, max_instances)):
        one_hot_mask[i] = instance_masks[i]

    return one_hot_mask

def save_decoded_images(decoded_data_dir, original, color_mask, instance_masks,
                        frame_id, video_name, json_file_name):
    # Save original and color masks
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'original/', video_name + '_' +
                 frame_id + '.jpg', original)
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'color_mask/', video_name + '_' +
                 frame_id + '.jpg', color_mask)

    # Save each instance mask separately
    # for category, masks in instance_masks.items():
    #     for i, mask in enumerate(masks):
    #         instance_filename = f"{video_name}_{frame_id}_{category}_instance_{i}.jpg"
    #         save_a_image(decoded_data_dir + json_file_name + '/' +
    #                      'instance_masks/', instance_filename, mask)

    # Save combined overlay of original and color masks
    alpha = 0.5
    overlay = cv2.addWeighted(original, 1 - alpha, color_mask, alpha, 0)
    original_mask_overlay = np.hstack((original, overlay))
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'original_plus_color_mask_overlay/',
                 video_name + '_' + frame_id +
                 '.jpg', original_mask_overlay)
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'color_mask_overlay/', video_name + '_'
                 + frame_id + '.jpg', overlay)


def create_mask(polygon_points, image_size=(100, 100),
                fill_color=255, outline_color=0):
    image = Image.new("L", image_size, outline_color)
    draw = ImageDraw.Draw(image)
    scaled_polygon = [(int(point['x'] * image_size[0]), int(point['y']
                       * image_size[1])) for point in polygon_points.values()]
    draw.polygon(scaled_polygon, fill=fill_color)
    return image


def get_specific_frame(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)
    frame_number = int(frame_id)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        print(video_path)
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return None
    cap.release()
    print(f"Frame {frame_number} extracted successfully.")
    return frame


def get_specific_frame_from_folder(video_path, frame_id):
    frame_number = int(frame_id)

    frame_filename = f"{frame_number:05d}.jpg"
    image_path= os.path.join(video_path, frame_filename) 
    frame = cv2.imread(image_path)
    return frame
def generate_one_hot_instance_mask_with_consistency(
    num_instances, H, W, instance_masks, instance_hashes, instance_dict
):
    """
    Generate a one-hot mask for up to 7 instances with temporal consistency.

    Args:
    - num_instances: Number of instances in the frame (maximum 7).
    - H: Height of the frame.
    - W: Width of the frame.
    - instance_masks: List of binary masks for each instance.
    - instance_hashes: List of unique hashes (objectHash) for each instance.
    - instance_dict: Dictionary tracking instance indices across frames.

    Returns:
    - one_hot_mask: A one-hot encoded mask of shape (7, H, W), where each
                    channel corresponds to an instance mask.
    """
    max_instances = 7
    one_hot_mask = np.zeros((max_instances, H, W), dtype=np.uint8)

    for i, instance_hash in enumerate(instance_hashes):
        if instance_hash not in instance_dict:
            if len(instance_dict) < max_instances:
                instance_dict[instance_hash] = len(instance_dict)  # Assign next available index
            else:
                print(f"Warning: More than {max_instances} instances in a frame; ignoring extra instances.")
                continue

        instance_index = instance_dict[instance_hash]
        one_hot_mask[instance_index] = instance_masks[i]

    return one_hot_mask
def generate_instance_color_mask_from_one_hot(one_hot_instances_mask):
    """
    Generate a color mask from a one-hot encoded instance mask.

    Args:
    - one_hot_instances_mask: A 3D numpy array of shape (H, W, 7) where each channel 
                              represents a different instance mask in one-hot encoding.

    Returns:
    - instance_color_mask: A color mask of shape (H, W, 3) where each instance channel 
                           is assigned a unique color.
    """
    # Define fixed colors for each channel
    colors = [
        (255, 0, 0),      # Red for channel 0
        (0, 0, 255),      # Blue for channel 1
        (0, 255, 0),      # Green for channel 2
        (255, 255, 0),    # Yellow for channel 3
        (255, 0, 255),    # Magenta for channel 4
        (0, 255, 255),    # Cyan for channel 5
        (128, 0, 128)     # Purple for channel 6
    ]
    
    # Initialize the instance color mask
    _,H, W = one_hot_instances_mask.shape
    instance_color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Apply colors for each instance channel
    for i in range(7):
        mask = one_hot_instances_mask[i,:, :]
        instance_color_mask[mask > 0] = colors[i]

    return instance_color_mask
def decode_json(json_data, json_file_name, categories,
                category_colors, raw_video_dir, decoded_data_dir,output_folder_pkl):
    decoded_data = json.loads(json_data)
    index = 0
    video_len =30
    for frame_data in decoded_data:
        labels = frame_data.get("data_units", {}).get(
            frame_data["data_hash"], {}).get("labels", {})
        H = frame_data.get("data_units", {}).get(
            frame_data["data_hash"], {}).get("height", {})
        W = frame_data.get("data_units", {}).get(
            frame_data["data_hash"], {}).get("width", {})
        image_size = (W, H)
        image_sizeR = (H, W)
        video_full_name = frame_data['data_title']
        video_name = video_full_name.split('.')[0]
        this_full_video_path = raw_video_dir + video_full_name
        this_seq_path  =  raw_video_dir + video_name + "/"
        print(video_full_name)
        print(this_full_video_path)
        video_images = []
        video_instance_masks = []
        instance_dict = {} 

        for id in range(video_len+1):
            key = str(id)
            this_frame_id = key
            # this_image = get_specific_frame(
            #     this_full_video_path, this_frame_id)
            this_image = get_specific_frame_from_folder(
                this_seq_path, this_frame_id)
            # this_seq_path
            max_instances = 7
            one_hot_instances_mask = np.zeros((max_instances, H, W), dtype=np.uint8)
            if key not in labels:
                print(f"Warning: Frame {key} does not have annotations.")
               
            # Get the specific frame from the video
            else:
                value = labels[key]
                
                this_label = value

                # Number of instances in this frame
                num_instances = len(this_label['objects'])

                # Generate unique colors for each instance
                instance_colors = generate_unique_colors(num_instances)

                # Semantic category masks (not instance specific)
                # category_masks = {category: np.zeros(
                #     image_sizeR, dtype=np.uint8) for category in categories}
                category_masks = {category: np.zeros(
                    image_sizeR, dtype=np.uint8) for category in categories}

                # Create an instance-level color mask
                instance_color_mask = np.zeros((H, W, 3), dtype=np.uint8)
                # instance_masks = []
                instance_masks = []
                instance_hashes = []
                for i, this_object in enumerate(this_label['objects']):
                    this_category_type = this_object['name']
                    this_object_hash = this_object['objectHash'] # read object hash

                    # Check if the object category is part of the defined categories
                    if this_category_type in categories:
                        this_polygon = this_object['polygon']
                        this_mask = create_mask(
                            this_polygon, image_size=image_size)
                        this_mask_array = np.array(this_mask)
                        # one_hot_instancesmask[i] = np.array(this_mask)

                        # Add the mask to the corresponding semantic category mask
                        category_masks[this_category_type] += this_mask_array

                        # Use the instance-specific color for this object
                        # instance_color_mask[this_mask_array > 0] = instance_colors[i]
                        instance_masks.append(np.array(this_mask))
                        instance_hashes.append(this_object_hash)
                one_hot_instances_mask = generate_one_hot_instance_mask_with_consistency(
                    num_instances, H, W, instance_masks, instance_hashes, instance_dict
                )
                instance_color_mask = generate_instance_color_mask_from_one_hot(one_hot_instances_mask)
                # Get the specific frame from the video
                # this_image = get_specific_frame(
                #     this_full_video_path, this_frame_id)
                
                if this_image is not None and num_instances > 0:
                    padded_frame_id = this_frame_id.zfill(4)
                    # one_hot_instance_mask = generate_one_hot_instance_mask(num_instances, H, W, instance_masks)

                    # Save the original frame, color mask (instance mask), and other data
                    save_decoded_images(decoded_data_dir,
                                        this_image,
                                        instance_color_mask,  # Save the instance color mask here
                                        category_masks,       # Save semantic masks per category
                                        padded_frame_id,
                                        video_name,
                                        json_file_name)
                    print(this_frame_id)
            # Add the image and mask to the video list
                    video_images.append(this_image)
                    video_instance_masks.append(one_hot_instances_mask)
        # for key, value in labels.items():
        #     this_frame_id = key
           
        video_images = np.array(video_images)
        video_instance_masks = np.array(video_instance_masks)
        video_images  = np.transpose(video_images , (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)
        video_instance_masks  = np.transpose(video_instance_masks , (1, 0, 2, 3))  # Reshape to (13, 29, 256, 256)
        print(video_images.shape)
        print(video_instance_masks.shape) 
        data_dict = {'frames': video_images,
                    'labels': video_instance_masks}


        pkl_file_name =  video_name + ".pkl"
        pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)

        with open(pkl_file_path, 'wb') as file:
            pickle.dump(data_dict, file)
            print("Pkl file created:" +pkl_file_name)
        index += 1
        print(index)


def main():
    jsonfile_dir = "C:/2data/training_data/selection_information/"
    json_to_decode = ['labels']
    # raw_video_dir = 'C:/2data/training_data/re_encoded_videos/'
    raw_video_dir = 'C:/2data/training_data/selected_video_frame_sequence/'

    decoded_data_dir = 'C:/2data/training_data/selected_GT/'
    output_folder_pkl = 'C:/2data/training_data/selected_GT/pkl/'
    categories =   [
        'Bipolar_dissector', #0
        'Bipolar_forceps', #1
        'Cadiere_forceps', #2
        'Clip_applier', #3
        'Force_bipolar', #4
        'Grasping_retractor', #5
        'Monopolar_curved_scissors', #6
        'Needle_driver', #7
        'Permanent_cautery_hook', #8
        'Prograsp_forceps', #9
        'Stapler', #10
        'Suction_irrigator', #11
        'Tip-up fenestrated_grasper', #12
        'Vessel_sealer' #13
    ]
    category_colors = {
        'Bipolar_dissector': (255, 0, 0),    # Red
        'Bipolar_forceps': (0, 255, 0),      # Green
        'Cadiere_forceps': (0, 0, 255),      # Blue
        'Clip_applier': (255, 128, 0),       # Orange
        'Force_bipolar': (128, 0, 128),      # Purple
        'Grasping_retractor': (0, 255, 255), # Cyan
        'Monopolar_curved_scissors': (255, 128, 128), # Light Red
        'Needle_driver': (128, 128, 0),      # Olive
        'Permanent_cautery_hook': (128, 0, 255), # Indigo
        'Prograsp_forceps': (0, 128, 128),   # Teal
        'Stapler': (255, 0, 128),            # Pink
        'Suction_irrigator': (128, 255, 0),  # Lime
        'Tip-up fenestrated_grasper': (255, 128, 0),    # Light Orange
        'Vessel_sealer': (0, 128, 255)       # Light Blue
    }

    for json_file_name in json_to_decode:
        with open(jsonfile_dir + json_file_name + '.json', 'r') as file:
            json_data = file.read()
        decode_json(json_data, json_file_name, categories,
                    category_colors, raw_video_dir, decoded_data_dir,output_folder_pkl)


if __name__ == '__main__':
    main()

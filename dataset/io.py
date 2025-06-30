import cv2
import numpy as np
import os
import random
# from matplotlib.pyplot import *
# # from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# import matplotlib.pyplot as plt
# # PythonETpackage for xml file edition
import pickle


def self_check_path_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_a_pkl(dir,name,object):
    with open(dir + name +'.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
    pass

def save_a_pkl_w_create(dir,name,object):
    self_check_path_create(dir)
    with open(dir + name +'.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
    pass

def read_a_pkl(dir, name):
  """
  Reads a pickled object from a file, handling potential missing files and encoding issues.

  Args:
      dir (str): The directory containing the pickle file.
      name (str): The name of the pickle file (without the .pkl extension).

  Returns:
      object: The loaded object from the pickle file, or None if the file is missing.
  """

  # Check if the file exists before attempting to load
  if not os.path.isfile(dir + name + '.pkl'):
    print(f"File '{dir + name + '.pkl'}' not found. Returning None.")
    return None

  try:
    # Attempt to load the object with 'iso-8859-1' encoding (optional)
    object = pickle.load(open(dir + name + '.pkl', 'rb'), encoding='iso-8859-1')
  except (EOFError, pickle.UnpicklingError) as e:
    print(f"Error loading pickle file '{dir + name + '.pkl'}: {e}. Returning None.")
    return None

  return object


def save_img_to_folder(this_save_dir,ID,img):
    # this_save_dir = Output_root + "1out_img/" + Model_key + "/ground_circ/"
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)
    cv2.imwrite(this_save_dir +
                str(ID) + ".jpg", img)
    
def save_a_image(dir,name,image):
    self_check_path_create(dir)
    cv2.imwrite(dir+name, image)


def load_a_video_buffer(video_path,video_buff_size,image_size,annotated_frame_ID,Display_loading_video ):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_down_sample = int((total_frames-1)/video_buff_size)
        # Read frames from the video clip
        frame_count = 0
        buffer_count = 0
        # Read frames from the video clip
        video_buffer = np.zeros((3, video_buff_size,   image_size,  image_size))
         
        frame_number =0
        Valid_video=False
        this_frame = 0
        previous_frame = 0
        previous_count =0
        while True:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if (frame_count %  video_down_sample==0):
                # start_time = time()

                ret, frame = cap.read()
      
                if ret == True:
                    H, W, _ = frame.shape
                  
                    
                     
                    this_resize = cv2.resize(frame, ( image_size,  image_size), interpolation=cv2.INTER_AREA)
                    reshaped = np.transpose(this_resize, (2, 0, 1))


                    # if frame_count %  video_down_sample==0:
                    video_buffer[:, buffer_count, :, :] = reshaped
                        
                    
                   
                    if buffer_count >=  video_buff_size:
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

        
       

        
        for annotated_id in annotated_frame_ID:
    # Set the position of the video capture object to the original frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, annotated_id)

            # Read the frame at the annotated frame index
            ret, frame = cap.read()

            if ret:
                H, W, _ = frame.shape
               

                this_resize = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                reshaped = np.transpose(this_resize, (2, 0, 1))

                # Calculate the corresponding index in the downsampled video buffer
                closest_frame_index = annotated_id * video_buff_size // total_frames
                closest_frame_index = min(closest_frame_index, video_buff_size - 1)

                # Replace the frame at the calculated index in the video buffer
                video_buffer[:, closest_frame_index, :, :] = reshaped
        cap.release()
        # return video_buffer, squeezed,Valid_video
        return video_buffer,Valid_video
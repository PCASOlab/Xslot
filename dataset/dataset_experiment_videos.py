import cv2
import os
import  numpy as np
from working_dir_root import Dataset_video_root, Dataset_label_root
import csv
import re
# idea record: If seperate the encoded left and right changels, then 14*2 out channels are required, and the flip data augmentation should swap the
#first 14 and last 14 channelds


categories = [
    'bipolar dissector',
    'bipolar forceps',
    'cadiere forceps',
    'clip applier',
    'force bipolar',
    'grasping retractor',
    'monopolar curved scissors',
    'needle driver',
    'permanent cautery hook/spatula',
    'prograsp forceps',
    'stapler',
    'suction irrigator',
    'tip-up fenestrated grasper',
    'vessel sealer'
]
# the list of labels
# Define the path to your CSV file
#############3 read all the labels##########################
csv_file_path = Dataset_label_root + "labels.csv"

# Initialize an empty list to store the data from the CSV file
data = []

# Open the CSV file and read its contents
try:
    with open(csv_file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Read the header row (if any)
        header = next(csvreader)

        # Read the remaining rows and append them to the 'data' list
        for row in csvreader:
            data.append(row)
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
    print(row)

labels = data
# conver label list into dictionary that can used key for fast lookingup
label_dict = {label_info[1]: label_info[2] for label_info in labels} # use the full name as the dictionary key
label_dict_number= {label_info[0]: label_info[2] for label_info in labels} # using the number and dictionary keey instead

#############3 read all the labels##########################
# Folder containing video clips
folder_path =  Dataset_video_root

# Number of clips to read
num_clips_to_read = 4  # Change this to the desired number of clips

# Initialize an array to store frames

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    frames_array = []

    if filename.endswith(".mp4"):
        # Extract clip ID from the filename
        clip_id = int(filename.split("_")[1].split(".")[0])
        clip_name = filename.split('.')[0]
        # label_Index = labels.index("clip_"+str(clip_id))
        # Check if the clip ID is within the range you want to read
        # if clip_id <= num_clips_to_read:
            # Construct the full path to the video clip
        video_path = os.path.join(folder_path, filename)
        this_label = label_dict [clip_name]
        binary_vector = np.array([1 if category in this_label else 0 for category in categories], dtype=int)
        # seperate the binary vector as left and right channel, so that when the image is fliped, two vector will exchange

        label_element  = re.findall(r'\w+(?:\s\w+)*|nan', this_label) # change to vector format instead of string

        # Initialize the label vector with 'nan' values
        label_vector = ['nan'] * 4  # Assuming a fixed length of 4 elements

        # Fill the label vector with category names
        for i, element in enumerate(label_element):
            label_vector[i] = element


        # label_vector_l = label_vector[0:2]
        # label_vector_r = label_vector[2:4]
        # binary_vector_l = np.array([1 if category in label_vector_l else 0 for category in categories], dtype=int)
        # binary_vector_r = np.array([1 if category in label_vector_r else 0 for category in categories], dtype=int)
        binary_vector_l = np.zeros(len(categories), dtype=int)
        binary_vector_r = np.zeros(len(categories), dtype=int)

        # Iterate through the label vector and set corresponding binary values
        for i in range (len( label_vector)):
            this_ele = label_vector[i]
            if this_ele in categories:
                category_index = categories.index(this_ele)
                # Determine if the category is in the left or right direction
                if i < (len(label_vector) /2):
                    binary_vector_l[category_index] = 1
                else:
                    binary_vector_r[category_index] = 1
        # Display the binary vector
        print(binary_vector)
        # Initialize a VideoCapture object
        cap = cv2.VideoCapture(video_path)

        # Read frames from the video clip
        frame_count = 0
        # Read frames from the video clip
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            x, y = 200, 100  # Position of the text

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (255, 255, 255)  # White color
            font_thickness = 2
            # Sample one frame per second (assuming original frame rate is 60 fps)
            if frame_count % (60*3) == 0:
                # frames_array.append(frame)
                cv2.putText(frame, this_label, (x, y), font, font_scale, font_color, font_thickness)

                cv2.imshow("First Frame", frame)
                cv2.waitKey(1)

            frame_count += 1

        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break

            # cv2.destroyAllWindows()

            # Store the frame in the frames_array
            # frames_array.append(frame)

        # Release the VideoCapture object
        cap.release()
        # Example: Display the first frame from the first clip
        # cv2.imshow("First Frame", frames_array[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# Now you have all the frames in frames_array and their corresponding clip IDs
# You can process the frames as needed
print(f"Total frames read: {len(frames_array)}")


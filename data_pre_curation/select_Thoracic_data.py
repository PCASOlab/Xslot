import os
import random
import shutil
import pickle

# Paths
source_folder = '/data/Thoracic/pkl/'
# destination_folder = r"C:\2data\training_data\selected"
selection_info_folder = '/data/Thoracic/selected/'

# Ensure destination and selection info folders exist
# os.makedirs(destination_folder, exist_ok=True)
os.makedirs(selection_info_folder, exist_ok=True)

# Get list of all mp4 files in the source folder
all_videos = [f for f in os.listdir(source_folder) if f.endswith('.pkl')]

# Randomly select 100 videos
selected_videos = random.sample(all_videos, 50)

# Move selected videos to the new folder
# for video in selected_videos:
#     shutil.copy(os.path.join(source_folder, video), os.path.join(destination_folder, video))

# Create a list of the unselected videos
unselected_videos = list(set(all_videos) - set(selected_videos))

# Save the selected and unselected lists to pkl files
with open(os.path.join(selection_info_folder, 'selected_videos.pkl'), 'wb') as f:
    pickle.dump(selected_videos, f)

with open(os.path.join(selection_info_folder, 'unselected_videos.pkl'), 'wb') as f:
    pickle.dump(unselected_videos, f)

print("Selection complete. Lists saved as .pkl files.")

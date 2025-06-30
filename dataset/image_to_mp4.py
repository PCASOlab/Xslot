import cv2
import os
import os
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai+thoracic'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai+cholec'  # Change this to your target mode


# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_endovis'  # Change this to your target mode
os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_miccai'  # Change this to your target mode

# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_cholec'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_thoracic'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_thoracic'  # Change this to your target mode



# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_cholec'  # Change this to your target mode

print("Current working directory:", os.getcwd())
from working_dir_root import Output_root,selected_data,Data_percentage

dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
Output_root = Output_root + "0Merge_predict" + dataset_tag + str(Data_percentage) + "/"

# Paths for input and output
match_color_mask_path = os.path.join(Output_root, "image/3 fps 0.9/match_color_mask")
match_color_mask_mp4_path = os.path.join(Output_root, "image/3 fps 0.9/match_color_mask_mp4")

# Ensure the output directory exists
os.makedirs(match_color_mask_mp4_path, exist_ok=True)

# Loop through each numbered subfolder (1, 2, 3, ...)
for number_folder in sorted(os.listdir(match_color_mask_path)):
    number_folder_path = os.path.join(match_color_mask_path, number_folder)

    if not os.path.isdir(number_folder_path) or not number_folder.isdigit():
        continue  # Skip non-directory or non-numeric folders

    print(f"Processing folder: {number_folder}")

    # Create corresponding output directory
    output_folder_path = os.path.join(match_color_mask_mp4_path, number_folder)
    os.makedirs(output_folder_path, exist_ok=True)

    # Loop through each subfolder inside the numbered folder
    for subfolder in os.listdir(number_folder_path):
        subfolder_path = os.path.join(number_folder_path, subfolder)

        if not os.path.isdir(subfolder_path):  # Ensure it's a directory
            continue

        images = [img for img in os.listdir(subfolder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        images = sorted(images)  # Sort naturally (frame_0, frame_1, etc.)

        if not images:
            print(f"No images found in {subfolder}")
            continue

        # Read the first image to get dimensions
        first_image = cv2.imread(os.path.join(subfolder_path, images[0]))
        height, width, layers = first_image.shape

        # Set FPS based on folder name
        fps = 3 if subfolder == "undownsampled_color" else 1

        # Define video writer with new output path
        video_path = os.path.join(output_folder_path, f"{subfolder}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write each image to the video
        for image in images:
            img_path = os.path.join(subfolder_path, image)
            frame = cv2.imread(img_path)
            out.write(frame)

        out.release()
        print(f"Video saved: {video_path} at {fps} FPS")
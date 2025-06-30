import cv2
import os

# Input and output directories
input_dir = '/media/guiqiu/Installation/database/feed back experiment/correct/no fd syn with a video/correct_orien/'
output_dir = '/media/guiqiu/Installation/database/feed back experiment/correct/no fd syn with a video/correct_orien_resize/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Desired size
new_size = (640, 640)

# List all files in the input directory
files = os.listdir(input_dir)

for file in files:
    # Check if the file is an image (you can add more image formats if needed)
    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
        try:
            # Read the image using OpenCV
            image = cv2.imread(os.path.join(input_dir, file))

            # Resize the image
            image = cv2.resize(image, new_size)

            # Save the resized image to the output directory
            output_path = os.path.join(output_dir, file)
            cv2.imwrite(output_path, image)

            print(f"Resized and saved: {file}")

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    else:
        print(f"Skipping {file} (not an image)")

print("All images processed.")

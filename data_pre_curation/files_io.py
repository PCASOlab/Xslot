import os
import cv2
import pickle

overwrite_all = False


def save_a_image(dir, name, image):
    self_check_path_create(dir)
    file_path = os.path.join(dir, name)
    if not should_overwrite(file_path):
        return
    cv2.imwrite(file_path, image)


def self_check_path_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_a_pkl(dir, name, obj):
    self_check_path_create(dir)
    file_path = os.path.join(dir, name + '.pkl')
    if not should_overwrite(file_path):
        return
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_a_pkl(dir, name):
    file_path = os.path.join(dir, name + '.pkl')
    obj = pickle.load(open(file_path, 'rb'), encoding='iso-8859-1')
    return obj


def should_overwrite(file_path):
    global overwrite_all
    if overwrite_all or not os.path.exists(file_path):
        return True
    while True:
        user_input = input(
            f"File {file_path} already exists. Overwrite? (no/yes/all): ").strip().lower()
        if user_input == 'no':
            return False
        elif user_input == 'yes':
            return True
        elif user_input == 'all':
            overwrite_all = True
            return True
        else:
            print("Invalid input. Please enter 'no', 'yes', or 'all'.")
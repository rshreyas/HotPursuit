from PIL import Image
import cv2
import os
import sys
from pprint import pprint
import numpy as np
from tqdm import tqdm

CWD = '/home/shreyas2/semester_3/mlmp_project/'
COUNTER = 0

def save_and_resize_single(path_to_images, img, square_size, save_dir_path):
    global COUNTER

    img = cv2.imread(os.path.join(path_to_images, img))
    img = cv2.resize(img, (square_size, square_size))
    cv2.imwrite(os.path.join(save_dir_path, str(COUNTER) + ".png") , img)
    COUNTER += 1


def pairwise_resize_and_join(path_to_images, color_img, thermal_img, square_size, save_dir_path):
    global COUNTER

    color_img = cv2.imread(os.path.join(path_to_images, color_img))
    color_img = cv2.resize(color_img, (square_size, square_size))

    thermal_img = cv2.imread(os.path.join(path_to_images, thermal_img))
    thermal_img = cv2.resize(thermal_img, (square_size, square_size))

    img = np.concatenate((color_img, thermal_img), axis=1)
    cv2.imwrite(os.path.join(save_dir_path, str(COUNTER) + ".png") , img)
    COUNTER += 1


def crop_images_and_save(path_to_images, save_dir_path, square_size, algo_name = 'pix2pix'):
    all_images = os.listdir(path_to_images)

    for image in all_images:
        if 'left_color' in image:
            left_color = image
        elif 'left_thermal' in image:
            left_thermal = image
        elif 'right_color' in image:
            right_color = image
        elif 'right_thermal' in image:
            right_thermal = image

    if algo_name == 'pix2pix':
        pairwise_resize_and_join(path_to_images, left_color, left_thermal, square_size, save_dir_path)
        pairwise_resize_and_join(path_to_images, right_color, right_thermal, square_size, save_dir_path)
    elif algo_name == 'cycleGAN':
        save_and_resize_single(path_to_images, left_color, square_size, save_dir_path)
        save_and_resize_single(path_to_images, left_thermal, square_size, save_dir_path)
        save_and_resize_single(path_to_images, right_color, square_size, save_dir_path)
        save_and_resize_single(path_to_images, right_thermal, square_size, save_dir_path)


for outdoor_indoor in tqdm(os.listdir(os.path.join(CWD, 'datasets', 'CATS_Release'))):
    for category in tqdm(os.listdir(os.path.join(CWD, 'datasets', 'CATS_Release', outdoor_indoor))):
        for scene in os.listdir(os.path.join(CWD, 'datasets', 'CATS_Release', outdoor_indoor, category)):
            for innermost_dir in os.listdir(os.path.join(CWD, 'datasets', 'CATS_Release', outdoor_indoor, category, scene)):
                # import pdb; pdb.set_trace()
                if innermost_dir == 'rawImages':
                    path_to_images = os.path.join(CWD, 'datasets', 'CATS_Release', outdoor_indoor, category, scene, innermost_dir)

                    algo_name = 'cycleGAN'
                    if not os.path.exists(os.path.join(CWD, 'datasets', 'joined_' + algo_name)):
                        os.makedirs(os.path.join(CWD, 'datasets', 'joined_' + algo_name))
                    crop_images_and_save(path_to_images, os.path.join(CWD, 'datasets', 'joined_' + algo_name), 256, algo_name)

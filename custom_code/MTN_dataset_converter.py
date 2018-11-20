from PIL import Image
import cv2
import os
import sys
from pprint import pprint
import numpy as np
from tqdm import tqdm

import pywt
import pywt.data

CWD = '/home/shreyas2/semester_3/mlmp_project/'
COUNTER = 0

def save_and_resize_single(img, square_size, save_dir_path):
    global COUNTER

    img = cv2.imread(img)
    img = cv2.resize(img, (square_size, square_size))
    cv2.imwrite(os.path.join(save_dir_path, str(COUNTER) + ".png") , img)
    COUNTER += 1


def add_hsv_to_rgb_img(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv


def add_wavelets_to_img(img, square_size):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    LL = cv2.resize(LL, (square_size, square_size))
    LH = cv2.resize(LL, (square_size, square_size))
    HL = cv2.resize(LL, (square_size, square_size))
    HH = cv2.resize(LL, (square_size, square_size))

    wavelets_out = np.concatenate((LL, LH, HL, HH), axis=-1)
    return wavelets_out


def add_external_features(image_path, square_size, hsv_required, wavelet_transform):
    hsv, waw = None, None

    image = cv2.imread(image_path)
    image = cv2.resize(image, (square_size, square_size))

    if hsv_required:
        hsv = add_hsv_to_rgb_img(image)

    if wavelet_transform:
        waw = add_wavelets_to_img(image, square_size)

    if hsv is not None:
        image = np.concatenate((image, hsv), axis=-1)

    if waw is not None:
        image = np.concatenate((image, waw), axis=-1)

    return image


def pairwise_resize_and_join(image_1_path, image_2_path, square_size, save_dir_path,
                             hsv_required=True, wavelet_transform=False):
    global COUNTER

    image_1 = add_external_features(image_1_path, square_size, hsv_required, wavelet_transform)
    image_2 = add_external_features(image_2_path, square_size, hsv_required, wavelet_transform)

    img = np.concatenate((image_1, image_2), axis=1)

    np.save(os.path.join(save_dir_path, str(COUNTER) + ".npy") , img)
    #cv2.imwrite(os.path.join(save_dir_path, str(COUNTER) + ".png") , img)

    COUNTER += 1


def crop_images_and_save(image_pairs, save_dir_path, square_size, algo_name = 'pix2pix'):
    '''
        pix2pix = Thermal, RGB
        pix2pix_rev = RGB, Thermal
        cycleGAN = single images
    '''

    for image_tuple in image_pairs:
        if algo_name == 'pix2pix':
            pairwise_resize_and_join(image_tuple[0], image_tuple[1], square_size, save_dir_path)
        elif algo_name == 'pix2pix_rev':
            pairwise_resize_and_join(image_tuple[1], image_tuple[0], square_size, save_dir_path)
        elif algo_name == 'cycleGAN':
            save_and_resize_single(image_tuple[0], square_size, save_dir_path)
            save_and_resize_single(image_tuple[1], square_size, save_dir_path)


for train_test in tqdm(os.listdir(os.path.join(CWD, 'datasets', 'MTN_DATASET'))):
    for outdoor_setting in tqdm(os.listdir(os.path.join(CWD, 'datasets', 'MTN_DATASET', train_test))):
        base_path = os.path.join(CWD, 'datasets', 'MTN_DATASET', train_test, outdoor_setting)
        list_of_left_images = os.listdir(os.path.join(base_path, 'LEFT'))
        base_file_names = [i.replace("LEFT", "") for i in list_of_left_images]

        thermal_left_pairs = [(base_path + "/THERMAL/THER" + i, base_path + "/LEFT/LEFT" + i) for i in base_file_names]
        thermal_right_pairs = [(base_path + "/THERMAL/THER" + i, base_path + "/RIGHT/RIGHT" + i) for i in base_file_names]

        algo_name = 'pix2pix'

        if not os.path.exists(os.path.join(CWD, 'datasets', 'joined_' + algo_name + '_MTN_A')):
            os.makedirs(os.path.join(CWD, 'datasets', 'joined_' + algo_name + '_MTN_A'))

        save_dir_path = os.path.join(CWD, 'datasets', 'joined_' + algo_name + '_MTN_A')
        crop_images_and_save(thermal_left_pairs, save_dir_path, 256, algo_name)
        crop_images_and_save(thermal_right_pairs, save_dir_path, 256, algo_name)

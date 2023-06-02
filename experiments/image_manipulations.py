import os
#import cv2
import random

'''
def occurrence_of_given_probability(probability):
    percentile = probability * 100
    sample = random.randint(0, 99)
    if sample <= percentile:
        return True
    else:
        return False


def read_img_from_path(name):
    return cv2.imread(name)


def rotate_image_randomly_90x_degrees(img):
    possible_rotate_codes = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    rotated_img = cv2.rotate(img, random.choice(possible_rotate_codes))
    return rotated_img


def add_gaussian_blurr(img):
    possible_kernels = [(11, 11), (5, 5), (3, 3), (7, 7), (13, 13)]
    blurred = cv2.GaussianBlur(img, random.choice(possible_kernels), 0)
    return blurred


def read_all_images_in_path_and_with_a_probability_rotate_and_save(path_to_folder, probability):
    os.chdir(path_to_folder)
    count_of_rotation = 0
    count = 0
    for image_file_name in os.listdir():
        image_incoming = read_img_from_path(image_file_name)
        count = count + 1
        if (image_incoming is not None) and (occurrence_of_given_probability(probability)):
            print("rotating")
            rotated = rotate_image_randomly_90x_degrees(image_incoming)
            cv2.imwrite("rotated_"+image_file_name, rotated)
            count_of_rotation = count_of_rotation + 1
    print("rotated "+str(count_of_rotation)+" images")
    print("in total "+str(count+count_of_rotation) + " images")


def read_all_images_in_path_and_with_a_probability_add_blur_and_save(path_to_folder, probability):
    os.chdir(path_to_folder)
    count_of_blur = 0
    count = 0
    for image_file_name in os.listdir():
        image_incoming = read_img_from_path(image_file_name)
        count = count + 1
        if (image_incoming is not None) and (occurrence_of_given_probability(probability)):
            print("adding blur")
            blurred = add_gaussian_blurr(image_incoming)
            cv2.imwrite("blurred_"+image_file_name, blurred)
            count_of_blur = count_of_blur + 1
    print("augmented "+str(count_of_blur)+" images")
    print("in total "+str(count+count_of_blur) + " images")
'''

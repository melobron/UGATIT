import os
import numpy as np
import cv2
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


################################# Path & Directory #################################
def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def make_exp_dir(main_dir):
    dirs = os.listdir(main_dir)
    dir_nums = []
    for dir in dirs:
        dir_num = int(dir[3:])
        dir_nums.append(dir_num)
    if len(dirs) == 0:
        new_dir_num = 1
    else:
        new_dir_num = max(dir_nums) + 1
    new_dir_name = 'exp{}'.format(new_dir_num)
    new_dir = os.path.join(main_dir, new_dir_name)
    return {'new_dir': new_dir, 'new_dir_num': new_dir_num}


################################# Model #################################


################################# Training #################################


################################# Transforms #################################
def get_transforms(args):
    transform_list = []
    if args.resize:
        transform_list.append(A.Resize(args.patch_size + 30, args.patch_size + 30))
    if args.crop:
        transform_list.append(A.RandomCrop(args.patch_size, args.patch_size))
    if args.flip:
        transform_list.append(A.HorizontalFlip(p=0.5))
    if args.normalize:
        transform_list.append(A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), max_pixel_value=255.0))
    transform_list.append(ToTensorV2())
    return transform_list


################################# ETC #################################
def tensor_to_numpy(tensor):
    img = tensor.mul(255).to(torch.uint8)
    img = img.numpy().transpose(1, 2, 0)
    return img


def denorm(tensor):
    return 0.5*(tensor + 1.0)


def RGB2BGR(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def cam(image, size=256):
    image = image - np.min(image)
    cam = image / np.max(image)
    cam = np.uint8(cam*255)
    cam = cv2.resize(cam, (size, size))
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return cam

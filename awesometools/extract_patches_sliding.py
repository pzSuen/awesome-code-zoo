from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import skimage.draw
import numpy as np
from tqdm.auto import tqdm
import cv2
import random

import warnings

warnings.filterwarnings('ignore')


def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def pad(img, pad_size=96):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 80 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    # print(img.shape, y_min_pad, y_max_pad, x_min_pad, x_max_pad)

    # 对于第3个channel大于512的情况，copyMakeBorder会报错（主要是细胞数量多时）
    if len(img.shape) == 3 and img.shape[2] > 512:
        image_list = []

        # 迭代的将channel放入list
        while img.shape[2] > 512:
            image_list.append(img[:, :, :512])
            img = img[:, :, 512:]

        image_list.append(img)
        image_pad_list = []
        # 分别 makeborder
        for image in image_list:
            pad_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
            if len(pad_image.shape) != len(image.shape):
                pad_image = pad_image[:, :, np.newaxis]
            image_pad_list.append(pad_image)
        # img = image_pad_list
        # 拼接回去
        img = np.concatenate(image_pad_list, 2)
    else:
        img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]


def read_nuclei(path):
    "read raw data"

    # Load 4-channel image
    img = skimage.io.imread(path)

    # input image
    if len(img.shape) > 2:
        img = img[:, :, :3]
    # mask
    else:
        # do nothing
        pass

    return img


def save_nuclei(path, img):
    "save image"
    skimage.io.imsave(path, img)


def sliding_window(image, step):
    x_loc = []
    y_loc = []
    # cells = []

    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            # cells.append(image[y:y + window[1], x:x + window[0]])
            x_loc.append(x)
            y_loc.append(y)
    return x_loc, y_loc


def extract_patches(image, step, patch_size):
    # print("*******************************")
    # print(image.shape)
    patches = []
    # Get locations
    x_pos, y_pos = sliding_window(image, step)
    for (x, y) in zip(x_pos, y_pos):
        # Get patch
        # patch = image[y:y + patch_size[0], x:x + patch_size[0]]
        patch = image[y:y + patch_size[0], x:x + patch_size[1]]

        # Get size
        # raw_dim = (patch.shape[1], patch.shape[0])  # W, H
        raw_dim = (patch.shape[0], patch.shape[1])  # W, H

        # print(raw_dim)
        # print(patch.shape)

        if raw_dim != (patch_size[0], patch_size[1]):

            # Resize to 64x64
            # patch = cv2.resize(patch, (64, 64), interpolation = cv2.INTER_AREA)
            old_shape = patch.shape
            patch, pad_locs = pad(patch, pad_size=patch_size[0])
            # print(old_shape, patch.shape)
            # 对于 h x w x 1, padding 后最后一个维度会消失
            if len(old_shape) != len(patch.shape):
                # print(old_shape, patch.shape)
                patch = patch[:, :, np.newaxis]

            # Do stuffffff
            patches.append(patch)

        else:
            # Do stuffffff
            patches.append(patch)

    patches = np.array(patches)

    return patches


# Sanity check

def show_image_mask_5class(image, mask):
    # Stolen from https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    norm = plt.Normalize(0, 4)  # 5 classes including BG
    map_name = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red", "yellow", "blue", "green"])
    h, w = mask.shape

    f, axarr = plt.subplots(1, 2, figsize=(16 * w / h * 2, 16))
    axarr[0].imshow(image, cmap=map_name, norm=norm)
    axarr[1].imshow(mask, cmap=map_name, norm=norm)
    plt.show()


if __name__ == "__main__":
    # label_map = {'Epithelial': 1,
    #              'Lymphocyte': 2,
    #              'Macrophage': 4,
    #              'Neutrophil': 3,
    #              }

    label_map = {'Epithelial': 1, 'Macrophage': 2, 'Neutrophil': 3, 'Lymphocyte': 4}
    # Root directory of the project
    ROOT_DIR = os.path.abspath("/home/pzsuen/MoNuSAC/")

    # Directory of images to run detection on
    IMAGES_SOURCE = os.path.join(ROOT_DIR, "image")
    MASKS_SOURCE = os.path.join(ROOT_DIR, "mask")
    CELLS_SOURCE = os.path.join(ROOT_DIR, "cell_masks")

    # Make new folders
    """
    Source File Tree:
        -- images
            -- TCGA-UZ-A9PO-01Z-00-DX1_1.png
            -- TCGA-UZ-A9PO-01Z-00-DX1_2.png
            -- ......
        -- masks
            -- TCGA-UZ-A9PO-01Z-00-DX1_1.png
            -- TCGA-UZ-A9PO-01Z-00-DX1_2.png
            -- ......
        -- cells
            -- TCGA-UZ-A9PO-01Z-00-DX1_1
                -- Epithelial
                    -- 1.png
                    -- 2.png
                    -- ......
                -- Macrophage
                -- Neutrophil
                -- Lymphocyte
            -- TCGA-UZ-A9PO-01Z-00-DX1_2
                -- Epithelial
                -- Macrophage
                -- Neutrophil
                -- Lymphocyte
    Destination File Tree:
        -- patches
            -- images
                -- TCGA-UZ-A9PO-01Z-00-DX1_1.png
                -- TCGA-UZ-A9PO-01Z-00-DX1_2.png
                -- ......
            -- masks
                -- TCGA-UZ-A9PO-01Z-00-DX1_1.png
                -- TCGA-UZ-A9PO-01Z-00-DX1_2.png
                -- ......
            -- cells
                -- TCGA-UZ-A9PO-01Z-00-DX1_1
                    -- Epithelial
                        -- 1.png
                        -- 2.png
                        -- ......
                    -- Macrophage
                    -- Neutrophil
                    -- Lymphocyte
                -- TCGA-UZ-A9PO-01Z-00-DX1_2
                    -- Epithelial
                    -- Macrophage
                    -- Neutrophil
                    -- Lymphocyte
        
    """
    IMAGES_DEST = os.path.join(ROOT_DIR, "patches", "images/")
    MASKS_DEST = os.path.join(ROOT_DIR, "patches", "masks/")
    CELLS_DEST = os.path.join(ROOT_DIR, "patches", "cells/")
    # Create destination folders
    create_directory(IMAGES_DEST)
    create_directory(MASKS_DEST)
    create_directory(MASKS_DEST)

    # Get all image file names
    image_fns = sorted(next(os.walk(IMAGES_SOURCE))[2])
    gt_fns = sorted(next(os.walk(MASKS_SOURCE))[2])

    assert len(image_fns) == len(gt_fns), "len(image_fns) != len(gt_fns)"

    # # Load a random image from the images folder
    # idx = random.randrange(len(image_fns))  # 94
    # print("Index: ", idx)
    #
    # image = skimage.io.imread(os.path.join(IMAGES_SOURCE, image_fns[idx]))
    # gt = skimage.io.imread(os.path.join(MASKS_SOURCE, gt_fns[idx]))
    #
    # assert image.shape[:2] == gt.shape, "Wrong image or ground truth!"
    # assert image.dtype == gt.dtype, "Wrong data types!"
    #
    # print(image.shape, gt.shape)
    #
    # val1 = gt.flatten()
    # print("Ground truth classes: ", np.unique(val1))
    # show_image_mask_5class(image, mask=gt)

    img_patches = None
    gt_patches = None

    # patch_size = (96, 96)
    # step = 16
    # img_patches = extract_patches(image, step, patch_size)
    # gt_patches = extract_patches(gt, step, patch_size)
    #
    # print('Patches shape: {}, {}'.format(img_patches.shape, gt_patches.shape))
    #
    # idxs = random.sample(range(0, len(img_patches)), 5)
    # # idxs = [random.randrange(len(img_patches)) for x in range(len(img_patches))]
    # print(idxs)
    # for img, msk in zip(img_patches[idxs], gt_patches[idxs]):
    #     print("Patch mask mean:", np.mean(msk))
    #
    #     # Set threshold for mask
    #     thresh = 0.1  # 0.099
    #     if np.mean(msk) < thresh:
    #         pass
    #
    #     else:
    #         show_image_mask_5class(img, msk)

    # Patch size and stride step
    patch_size = (256, 256)
    step = 128
    threshold = 0.1

    # Iterate over all image and masks
    ct = 0

    for img_path, gt_path in tqdm(zip(image_fns[:], gt_fns[:])):
        # print(img_path, gt_path)
        # img_path = 'TCGA-A2-A0ES-01Z-00-DX1_2.png'
        # gt_path = 'TCGA-A2-A0ES-01Z-00-DX1_2.png'
        patient_case_id = img_path.split('.')[0]
        print(patient_case_id)
        if len(patient_case_id.split('_')) == 2:  # TCGA-55-1594-01Z-00-DX1_001.png
            patient_id = patient_case_id.split('_')[0]
            case_id = patient_case_id.split('_')[1]
        else:  # TCGA-55-1594-01Z-00-DX1-001.png
            patient_id = '-'.join(patient_case_id.split('-')[:-1])
            case_i = patient_case_id.split('-')[-1]

        cell_root = os.path.join(CELLS_SOURCE, patient_id, patient_case_id)
        # if len(os.listdir(cell_root)) != 4:
        #     print(patient_case_id)

        cell_classes = ('Epithelial', 'Macrophage', 'Neutrophil', 'Lymphocyte')
        all_cell_dict = dict()
        # 遍历每个类别
        for cls in cell_classes:
            cell_path = os.path.join(cell_root, cls)
            if os.path.exists(cell_path):
                cell_fnames = glob(os.path.join(cell_path, '*.png'))
                if len(cell_fnames) > 0:
                    all_cells = []
                    # 遍历该类别下所有细胞
                    for cf in cell_fnames:
                        cell = read_nuclei(cf)
                        all_cells.append(cell)
                    all_cells = np.array(all_cells)

                    all_cell_dict[cls] = all_cells
            # else:
            #     all_cell_dict[cls] = None
        # print('\n$##########################################')
        # print(patient_case_id)
        # for k, v in all_cell_dict.items():
        #     print(k, v.shape)

        # Read image and ground truth
        image = skimage.io.imread(os.path.join(IMAGES_SOURCE, img_path))
        gt = skimage.io.imread(os.path.join(MASKS_SOURCE, gt_path))

        # # Extract patches
        img_patches = extract_patches(image, step, patch_size)
        gt_patches = extract_patches(gt, step, patch_size)
        # print(img_patches.shape)
        # print(gt_patches.shape)
        # print("############################################")
        # print(img_patches.shape, gt_patches.shape)
        cell_patches_dict = dict()
        for k, v in all_cell_dict.items():
            # print(k, v.shape)
            # cell_patches = None
            v = v.transpose((1, 2, 0))
            # print(v.shape)
            cell_patches = extract_patches(v, step, patch_size)
            # print(cell_patches.shape)
            cell_patches_dict[k] = cell_patches
            # print(cell_patches.shape)
        # print("############################################")

        # if len(gt_patches) == 0:
        #     print("Not included")

        for i in range(len(img_patches)):
            # for im, msk,cpd in zip(img_patches, gt_patches,cell_patches_dict):
            im = img_patches[i]
            msk = gt_patches[i]

            # Threshold
            if np.mean(msk) < threshold:
                pass

            else:
                # show image and mask with a certain probability
                # if np.random.rand(1) < 1:
                #     show_image_mask_5class(im, msk)

                # Save image patch
                save_nuclei(IMAGES_DEST + "{}.png".format(ct), im)
                # Save mask patch
                save_nuclei(MASKS_DEST + "{}.png".format(ct), msk)
                # Save cell patch

                for k, v in cell_patches_dict.items():
                    this_cell_path = os.path.join(CELLS_DEST, str(ct), k)
                    # print(v.shape)
                    cell_masks = v[i]
                    cell_masks = cell_masks.transpose(2, 0, 1)
                    ci = 1
                    for cm in cell_masks:
                        if np.sum(cm) > 0:
                            create_directory(this_cell_path)
                            save_nuclei(os.path.join(this_cell_path, str(ci) + '.png'), cm)
                            ci += 1
                    # create_directory(os.path.join(CELLS_DEST, str(ct), '1'.png))

                ct += 1

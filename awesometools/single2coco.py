import os
import cv2
import glob
import skimage.io as io
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# import sklearn
# import shutil
import matplotlib.pyplot as plt


def convert2coco(image_fnames, is_train=True):
    """ convert every single cells of one image to coco """
    # 总的
    CELL_CLASSES = {'Epithelial': 1, 'Macrophage': 2, 'Neutrophil': 3, 'Lymphocyte': 4}

    coco = {
        "info": [],
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "supercategory": "cell",
                "id": 1,
                "name": "Epithelial"
            },
            {
                "supercategory": "cell",
                "id": 2,
                "name": "Macrophage"
            },
            {
                "supercategory": "cell",
                "id": 3,
                "name": "Neutrophil"
            },
            {
                "supercategory": "cell",
                "id": 4,
                "name": "Lymphocyte"
            },
        ]
    }

    curr_image = {
        "file_name": [],
        "height": 480,
        "width": 480,
        "id": []
    }
    images = []

    curr_annotation = {
        "segmentation": [],
        "area": [],
        "iscrowd": 0,
        "image_id": [],
        "bbox": [],
        "category_id": 0,
        "id": []
    }
    annotations = []

    anno_id = 0
    for image_fname in tqdm(sorted(image_fnames)):
        # print(image_fname)
        image_id = image_fname.split('.')[0]
        # shutil.copy(image_path, os.path.join('/home/pzsuen/Code/PolarMask2/DSB2018_Fluorescence/images/', id + '.png'))
        image = cv2.imread(os.path.join(IMAGE_PATH, image_fname))
        cell_root = os.path.join(ROOT, "cells", image_id)

        # add image info
        h, w, _ = image.shape
        curr_image["file_name"] = image_fname
        curr_image["id"] = image_id
        curr_image["height"] = h
        curr_image["width"] = w
        images.append(curr_image)

        update_contours = []
        update_bboxs = []
        update_areas = []
        update_category = []

        curr_image_classes = os.listdir(cell_root)
        for cic in curr_image_classes:  # 所有类别
            cell_path = os.path.join(cell_root, cic)
            cell_fps = glob.glob(os.path.join(cell_path, '*.png'))
            for cfp in cell_fps:  # 该类别所有细胞
                mask = cv2.imread(cfp, 0)
                # print(np.unique(mask), mask.shape)
                # plt.subplot(121)
                # plt.imshow(image)
                # plt.subplot(122)
                # plt.imshow(mask,alpha=0.5)
                # plt.show()
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour = contours[0]
                x1 = np.min(contour[:, :, 0])
                y1 = np.min(contour[:, :, 1])
                x2 = np.max(contour[:, :, 0])
                y2 = np.max(contour[:, :, 1])
                if (x2 - x1) == 0 or (y2 - y1) == 0:
                    continue

                update_bboxs.append([x1, y1, (x2 - x1), (y2 - y1)])
                update_areas.append((x2 - x1) * (y2 - y1))
                update_contours.append(contour.squeeze(1).flatten().tolist())
                update_category.append(CELL_CLASSES[cic])

        for (seg, bbox, area, cate) in zip(update_contours, update_bboxs, update_areas, update_category):
            curr_annotation["segmentation"] = [seg]
            curr_annotation["bbox"] = list(map(int, bbox))
            curr_annotation["area"] = str(area)
            curr_annotation["image_id"] = image_id
            curr_annotation["id"] = str(anno_id)
            curr_annotation["category_id"] = cate
            annotations.append(curr_annotation)
            curr_annotation = curr_annotation.copy()
            anno_id += 1

        curr_image = curr_image.copy()
        # image_id += 1

    coco["images"] = images
    coco["annotations"] = annotations

    print(coco)
    # with open("/home/tikboa/306_server_work/DSB2018/annotations/val.json", 'w') as f:
    print("There are are {} cells.".format(anno_id))
    if is_train:
        with open(os.path.join(ROOT, "train.json"), 'w') as f:
            json.dump(coco, f)
    else:
        with open(os.path.join(ROOT, "val.json"), 'w') as f:
            json.dump(coco, f)


if __name__ == '__main__':
    # TRAIN_PATH = "/home/tikboa/306_server_work/cell_segmentation_dataset/DSB2018/stage1_train/"
    # TRAIN_PATH = "/home/pzsuen/Code/PolarMask2/DSB2018_Fluorescence/data/"
    ROOT = "/home/pzsuen/MoNuSAC/patches/"
    IMAGE_PATH = os.path.join(ROOT, 'images')
    # CELL_PATH = os.path.join(ROOT, 'cells')

    image_fnames = os.listdir(IMAGE_PATH)  # next(os.walk(IMAGE_PATH))[1]
    train_fnames, val_fnames, _, _ = train_test_split(image_fnames, range(len(image_fnames)), test_size=0.02,
                                                      random_state=2020,
                                                      shuffle=True)
    print(len(image_fnames), len(train_fnames), len(val_fnames))

    # convert2coco(train_fnames, True)
    convert2coco(val_fnames, False)

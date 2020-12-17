import cv2, os
import numpy as np
import time
from glob import glob
from imageio import imread
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def image_mean_std(root_path, image_format='.png', CNum=0):
    '''
    计算数据集中图片的均值和方差
    :param root_path: 所有图片所在的根目录
    :param image_format: 图像格式，如 .png,.jpeg等，注意：如果mask文件在同一根目录下且格式相同，在glob中要利用正则表达式进行过滤
    :param CNum: 读取多少图像进行计算，0 for all.
    :return: mean and std
    '''
    # CNum = 100  # 挑选多少图片进行计算

    image_fname_list = glob(os.path.join(root_path, "*_?" + image_format))
    print(image_fname_list)
    print(len(image_fname_list))
    images = []
    print("Loading the images ......")

    pixel_num_sum = 0
    pixel_mean_sum = 0
    for i in range(image_fname_list.__len__()):
        image_fname = image_fname_list[i]
        # img = cv2.imread(image_fname) / 255.
        img = imread(image_fname) / 255.
        # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        pixel_mean_sum = np.mean(img, axis=(0, 1))
        # print(pixel_num_sum.shape)
        pixel_num_sum += h * w

    pixel_std_sum = 0
    for i in range(image_fname_list.__len__()):
        image_fname = image_fname_list[i]
        # img = cv2.imread(image_fname) / 255.
        img = imread(image_fname) / 255.
        # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel_std_sum += np.sum((img - pixel_mean_sum[np.newaxis, np.newaxis, :]) ** 2, axis=(0, 1))

    pixel_std_sum /= pixel_num_sum
    pixel_std_sum = np.sqrt(pixel_std_sum)

    print("normMean = {}".format(pixel_mean_sum))
    print("normStd = {}".format(pixel_std_sum))


if __name__ == '__main__':
    start = time.time()
    root_path = '/Share2/HuBMAP/cut_cortex/All'
    image_mean_std(root_path)
    end=time.time()

    print(f"Using {(end-start)/60} minutes......")

# [0.45386231 0.32667477 0.5053146 ]

# normMean = [0.45386231 0.32667477 0.5053146 ]
# normStd = [ 7812128.35666658  5988768.45946425 11215957.08400804]


import os, json, cv2, csv, time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from imageio import imread, imwrite
from skimage import draw


def sliding_window(image, mask=None, image_id='0', scales=3, base_window_size=256, step=256,
                   pad_mod='symmetric', save_root='../data/patches', filter_ratio_threshold=0.7):
    '''
    文件命名规则：
        目录结构：
            -- Patches_{scales}_{base_window_size}_{step}
                -- image
                    {ImageId}_{PatchID}_{row_id}_{column_id}_{scale_at}.png
                    ...
                -- mask
                    {ImageId}_{PatchID}_{row_id}_{column_id}_{scale_at}.png
                    ...


    :param image: np.array, size=(height,width,3)
    :param mask: np.array, size=(height,width) ; or it is None
    :param image_id: such as 1e2425f28
    :param cortex_id: cortex id in one image
    :param scales: a integer from [1,2,3], how much scales do you want
    :param base_window_size: the minimal window size, which needs to be divided by 2
    :param step: int
    :param pad_mod: padding mod which is the same as np.pad()
    :param save_root:
    :return: save image (mask) to save_dist
    '''
    height, width, _ = image.shape
    assert base_window_size <= height and base_window_size <= width, "The image is too small to split."
    assert scales in [1, 2, 3], "Only 1,2,3 is okay for scale."
    assert base_window_size % 2 == 0, "Base window size needs to be divisible by 2."

    # 创建存储目录
    save_dist = os.path.join(save_root, f'patches_{scales}_{base_window_size}_{step}')
    if not os.path.exists(save_dist):
        os.makedirs(os.path.join(save_dist, 'mask'))
        os.makedirs(os.path.join(save_dist, 'image'))

    # 设置各尺度窗口大小
    if scales == 3:
        out_window_size = base_window_size * 4
        mid_window_size = base_window_size * 2
        max_window_size = out_window_size
    elif scales == 2:
        mid_window_size = base_window_size * 2
        max_window_size = mid_window_size
    else:
        max_window_size = base_window_size

    # 基础padding大小
    pad_length = (max_window_size - base_window_size) // 2
    # 原大小不能整除base_window_size，需要再进行padding
    height_left = height % base_window_size
    width_left = width % base_window_size
    height_add_pad = base_window_size - height_left
    width_add_pad = base_window_size - width_left

    # 对图像进行padding
    image_pad = np.pad(array=image,
                       pad_width=(
                           (pad_length, pad_length + height_add_pad), (pad_length, pad_length + width_add_pad), (0, 0)),
                       mode=pad_mod)
    if mask is not None:
        mask_pad = np.pad(array=mask, pad_width=(
            (pad_length, pad_length + height_add_pad), (pad_length, pad_length + width_add_pad)),
                          mode=pad_mod)

    # 计算共计多少个patch
    # all_num = (height + height_add_pad) * (width + width_add_pad) // base_window_size ** 2
    # print("All number: ", all_num * 6)

    # 进行裁剪
    new_height, new_width, _ = image_pad.shape
    patch_id = 0
    row_id = 0
    for lty in range(pad_length, new_height - pad_length, step):  # left top y axis
        row_id += 1
        column_id = 0
        for ltx in range(pad_length, new_width - pad_length, step):  # left top x axis
            column_id += 1
            patch_id += 1

            # 裁剪base window
            scale_at = 1
            base_window_image = image_pad[lty:lty + base_window_size, ltx:ltx + base_window_size, :]

            if not filter_empty_patches(base_window_image, frh=filter_ratio_threshold):
                continue

            cv2.imwrite(
                os.path.join(save_dist, 'image',
                             f'{image_id}_{patch_id}_{row_id}_{column_id}_{scale_at}.png'),
                base_window_image)
            if mask is not None:
                base_window_mask = mask_pad[lty:lty + base_window_size, ltx:ltx + base_window_size]
                cv2.imwrite(
                    os.path.join(save_dist, 'mask',
                                 f'{image_id}_{patch_id}_{row_id}_{column_id}_{scale_at}.png'),
                    base_window_mask)

            # 进行多尺度窗口裁剪
            if scales >= 2:
                # 进行第3尺度裁剪
                if scales == 3:
                    bod = (out_window_size - base_window_size) // 2  # base_out_distance
                    out_window_image = image_pad[lty - bod:lty - bod + out_window_size,
                                       ltx - bod:ltx - bod + out_window_size, :]

                    scale_at = 3
                    cv2.imwrite(
                        os.path.join(save_dist, 'image',
                                     f'{image_id}_{patch_id}_{row_id}_{column_id}_{scale_at}.png'),
                        out_window_image)

                    if mask is not None:
                        out_window_mask = mask_pad[lty - bod:lty - bod + out_window_size,
                                          ltx - bod:ltx - bod + out_window_size]
                        cv2.imwrite(
                            os.path.join(save_dist, 'mask',
                                         f'{image_id}_{patch_id}_{row_id}_{column_id}_{scale_at}.png'),
                            out_window_mask)

                # 进行第2尺度裁剪
                scale_at = 2
                bmd = (mid_window_size - base_window_size) // 2  # base_mid_distance
                mid_window_image = image_pad[lty - bmd:lty - bmd + mid_window_size,
                                   ltx - bmd:ltx - bmd + mid_window_size, :]
                cv2.imwrite(
                    os.path.join(save_dist, 'image',
                                 f'{image_id}_{patch_id}_{row_id}_{column_id}_{scale_at}.png'),
                    mid_window_image)

                if mask is not None:
                    mid_window_mask = mask_pad[lty - bmd:lty - bmd + mid_window_size,
                                      ltx - bmd:ltx - bmd + mid_window_size]

                    cv2.imwrite(
                        os.path.join(save_dist, 'mask',
                                     f'{image_id}_{patch_id}_{row_id}_{column_id}_{scale_at}.png'),
                        mid_window_mask)


def filter_empty_patches(patch, frh):
    '''
    如果非空白区域面积小于frh，返回flase; 否则，返回true
    :param patch: shape=(h,w,3)
    :param frh: filter_ratio_threshold, the useful region ratio bigger than it.
    :return:
    '''
    patch = np.average(patch, axis=2)
    patch = (patch > 0) * 1
    white = np.sum(patch, axis=(0, 1))
    black = patch.shape[0] * patch.shape[1]
    white_ratio = white / black

    # print(white_ratio)
    if white_ratio < frh:
        return False
    else:
        return True


if __name__ == '__main__':
    s = time.time()
    data_root = '/Share2/HuBMAP/cut_cortex/train/'
    img_fname_list = glob(os.path.join(data_root, "*_?.png"))
    # img_fname_list = ['/home/pzsuen/Code/HuBMAP/data/e79de561c_0.png']
    for img_fname in tqdm(img_fname_list):
        basedir = os.path.dirname(img_fname)
        full_id = os.path.basename(img_fname).split('.')[0]
        image_id = full_id.split('_')[0]
        image = cv2.imread(img_fname)
        mask = cv2.imread(os.path.join(basedir, full_id + '_mask.png'), cv2.IMREAD_GRAYSCALE)

        sliding_window(image=image, mask=mask, image_id=image_id, scales=3, base_window_size=256,
                       step=256, pad_mod='symmetric', save_root='../data/', filter_ratio_threshold=0.7)

    e = time.time()

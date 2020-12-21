from imageio import imread,imsave
from awesometools.color_norm.color_normalize import reinhard_normalizer

def color_normalize(ref_image_fname, target_image_fname,save_dist = None):
    rn = reinhard_normalizer(ref_image_fname)  # target_40X.png')

    target_image = imread(target_image_fname)
    img_normalized = rn.normalize(target_image)

    if save_dist is not None:
        imsave(save_dist,img_normalized)
    return img_normalized


if __name__ == '__main__':
    ref_image_fname = './color_norm/TCGA-SX-A7SR-01Z-00-DX1_2.png'
    target_image_fname = './color_norm/TCGA-P4-AAVK-01Z-00-DX1_2.png'
    _ = color_normalize(ref_image_fname,target_image_fname,save_dist='./color_norm/TCGA-P4-AAVK-01Z-00-DX1_2_normalized.png')
    

from imageio import imread, imsave
from awesometools.color_norm.color_normalize import reinhard_normalizer
import staintools, cv2
from imageio import imread
import matplotlib.pyplot as plt


def color_normalize(ref_image_fname, target_image_fname, save_dist=None):
    rn = reinhard_normalizer(ref_image_fname)  # target_40X.png')

    target_image = imread(target_image_fname)
    img_normalized = rn.normalize(target_image)

    if save_dist is not None:
        imsave(save_dist, img_normalized)
    return img_normalized


if __name__ == '__main__':
    ref_image_fname = '../images/TCGA-SX-A7SR-01Z-00-DX1_2.png'
    target_image_fname = '../images/TCGA-P4-AAVK-01Z-00-DX1_2.png'
    _ = color_normalize(ref_image_fname, ref_image_fname,
                        save_dist='../images/TCGA-SX-A7SR-01Z-00-DX1_2_normalized.png')
    print('+++++++++++++++')
    norm = staintools.reinhard_color_normalizer.ReinhardColorNormalizer()
    ta = cv2.cvtColor(cv2.imread(ref_image_fname), cv2.COLOR_BGR2RGB)

    ref = cv2.cvtColor(cv2.imread(target_image_fname), cv2.COLOR_BGR2RGB)
    norm.fit(target=ta)
    img = norm.transform(I=ref)
    img_ta = norm.transform(I=ta)
    plt.subplot(221)
    plt.imshow(ta)
    plt.subplot(222)
    plt.imshow(img_ta)
    plt.subplot(223)
    plt.imshow(ref)
    plt.subplot(224)
    plt.imshow(img)

    plt.show()

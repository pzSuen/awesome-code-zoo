import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


# todo:理解该代码逻辑
def get_edges_4(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def get_edges_2(t):
    # edge = torch.ByteTensor(t.size()).zero_()
    edge = np.zeros_like(t)
    edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
    edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
    edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
    edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
    return edge.astype(float)

def draw_contour(image, mask):
    '''
    :param image: cv.IMREAD_COLOR
    :param mask: cv.IMREAD_GRAYSCALE
    :return:
    '''
    countour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))

    cv2.drawContours(image, countour, -1, (0, 255, 0), 3)
    return image

if __name__ == "__main__":
    import time

    time_start = time.time()

    path = '../images/TCGA-73-4668-01Z-00-DX1_004.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    tm = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    edge = get_edges_4(tm).squeeze()
    plt.imshow(edge)
    print(edge.shape)
    # print(edge.shape)
    plt.imsave("./TCGA-73-4668-01Z-00-DX1_004_edge.png", edge)
    plt.show()

    time_end = time.time()
    print('totally cost', time_end - time_start)
    print(time.time())

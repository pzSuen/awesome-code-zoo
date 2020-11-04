import cv2
import torch
import matplotlib.pyplot as plt


# todo:理解该代码逻辑
def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


if __name__ == "__main__":
    path = './TCGA-73-4668-01Z-00-DX1_004.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    tm = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    edge = get_edges(tm).squeeze()
    plt.imshow(edge)
    # print(edge.shape)
    plt.imsave("./TCGA-73-4668-01Z-00-DX1_004_edge.png",edge)
    plt.show()

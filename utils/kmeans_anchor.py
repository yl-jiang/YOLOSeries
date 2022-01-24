import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import torch
from sklearn.cluster import KMeans

def show_result(class_list, raw_data, center_coordinate):
    print('Showing... ...')
    colors = [
        '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
        '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
        '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]

    use_color = []
    for node in class_list:
        use_color.append(colors[int(node)])

    plt.figure(num=1, figsize=(16, 9))
    plt.scatter(x=raw_data[:, 0], y=raw_data[:, 1], c=use_color, s=50, marker='o', alpha=0.3)
    plt.scatter(x=center_coordinate[:, 0], y=center_coordinate[:, 1], c='b', s=200, marker='+', alpha=0.8)
    
    plt.title("COCO KMeans Anchors", fontdict={'weight': 'bold', 'fontsize': 14})
    plt.xlabel('Width', fontdict={'weight': 'bold', 'fontsize': 14})
    plt.ylabel('Height', fontdict={'weight': 'bold', 'fontsize': 14})
    plt.savefig("./utils/kmeans_anchors.jpg", dpi=250)
    plt.show()


def skkmeans(data, n_clusters):
    """
    Return kmeans anchors and save figure of clusters.
    Args:
        data: [[box_width_norm, box_height_norm]] matrix (N, 2) ndarray;
        n_clusters: anchor numbers
    """
    kmean = KMeans(n_clusters=n_clusters).fit(data)
    labels_ = kmean.labels_
    centers_ = kmean.cluster_centers_
    show_result(labels_, data, centers_)
    return centers_

# --------------------------------------------- RetinaNet Anchor ---------------------------------------------

def compute_featuremap_shape(img_shape, pyramid_level):
    """
    compute feature map's shape based on pyramid level.
    :param img_shape: 3 dimension / [h, w, c]
    :param pyramid_level: int
    :return:
    """
    img_shape = np.array(img_shape)
    fm_shape = (img_shape - 1) // (2**pyramid_level) + 1
    return fm_shape


class CPUAnchor:
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]

        if ratios is None:
            # the ratio of anchor's height / anchor's width
            self.ratios = np.array([0.5, 1, 2])

        if scales is None:
            # the ratio of anchors areas
            self.scales = np.array([1, 2 ** (1 / 3), 2 ** (2 / 3)])

    def __call__(self, img_shape):
        anchor_output = np.zeros((0, 4), dtype=np.float32)

        for i in range(len(self.pyramid_levels)):
            fm_shape = compute_featuremap_shape(img_shape, self.pyramid_levels[i])
            base_anchor = self._base_anchor_generator(self.sizes[i])
            base_anchor_shifted = self._shift(fm_shape, self.strides[i], base_anchor)
            anchor_output = np.append(anchor_output, base_anchor_shifted, axis=0)

        return anchor_output

    def _base_anchor_generator(self, anchor_size):
        anchor_num = len(self.scales) * len(self.ratios)
        base_anchors = np.zeros([anchor_num, 4])

        # shape: (9, 2) | (3,) -> (2, 9) -> (9, 2)
        base_anchors[:, 2:] = anchor_size * np.tile(self.scales, (2, len(self.ratios))).T
        areas = base_anchors[:, 2] * base_anchors[:, 3]

        base_anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        base_anchors[:, 3] = base_anchors[:, 2] * np.repeat(self.ratios, len(self.scales))

        base_anchors[:, 0::2] -= np.tile(base_anchors[:, 2], (2, 1)).T * 0.5
        base_anchors[:, 1::2] -= np.tile(base_anchors[:, 3], (2, 1)).T * 0.5

        return base_anchors

    def _shift(self, shape, stride, anchors):
        """
        增加anchor的多样性
        :param shape:
        :param stride:
        :param anchors: format->(-w/2, -h/2, w/2, h/2)
        :return:
        """
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # shape: (K, 4)
        shifts = np.vstack([shift_x.ravel(),
                            shift_y.ravel(),
                            shift_x.ravel(),
                            shift_y.ravel()]).transpose()
        # [K, 4] -> [1, K, 4] -> [K, 1, 4]
        shifts = shifts[None, :, :].transpose((1, 0, 2))
        # [K, 1, 4] & [1, A, 4] -> [K, A, 4]
        all_anchors = shifts + anchors[None, :, :]
        # [K, A, 4] -> [K*A, 4]
        return np.ascontiguousarray(all_anchors.reshape(-1, 4))


class GPUAnchor:
    def __init__(self,
                 input_img_size,
                 pyramid_levels=None,
                 strides=None,
                 sizes=None,
                 ratios=None,
                 scales=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_shape = input_img_size

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        if sizes is None:
            # 2**(x+2):因为ResidualNet head对图像进行了4倍下采样
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]

        if ratios is None:
            # the ratio of anchor's height / anchor's width
            self.ratios = torch.tensor([0.5, 1, 2], device=self.device)

        if scales is None:
            # the ratio of anchors areas
            self.scales = torch.tensor([1, 2 ** (1 / 3), 2 ** (2 / 3)], device=self.device)

    def __call__(self):
        """
        compute all anchors according to all pyramid levels.
        Return:
            anchors: (N, 4)
        """
        assert len(self.img_shape) == 2
        anchor_output = []

        for i in range(len(self.pyramid_levels)):
            fm_shape = self.compute_featuremap_shape(self.img_shape, self.pyramid_levels[i])
            base_anchor = self._base_anchor_generator(self.sizes[i])
            base_anchor_shifted = self._shift(fm_shape, self.strides[i], base_anchor)
            anchor_output.append(base_anchor_shifted)

        return torch.cat(anchor_output, dim=0).contiguous()

    def _base_anchor_generator(self, size):
        anchor_num = self.scales.nelement() * self.ratios.nelement()
        base_anchors = torch.zeros(anchor_num, 4, device=self.device)

        # shape: (9, 2)
        base_anchors[:, 2:] = size * self.scales.repeat(2, len(self.ratios)).transpose(0, 1)
        areas = base_anchors[:, 2] * base_anchors[:, 3]

        base_anchors[:, 2] = torch.sqrt(areas / self.ratios.repeat_interleave(self.scales.nelement()))
        base_anchors[:, 3] = base_anchors[:, 2] * self.ratios.repeat_interleave(len(self.scales))

        # [xmin, ymin, xmax, ymax]: [-w/2, -h/2, w/2, h/2]
        base_anchors[:, 0::2] -= base_anchors[:, 2].repeat(2, 1).T / 2
        base_anchors[:, 1::2] -= base_anchors[:, 3].repeat(2, 1).T / 2
        return base_anchors.contiguous()

    def _shift(self, shape, stride, anchors):
        """
        shift anchors.
        :return:
        """
        shift_x = (torch.arange(0, shape[1], device=self.device) + 0.5) * stride
        shift_y = (torch.arange(0, shape[0], device=self.device) + 0.5) * stride
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)

        # shape: (K, 4)
        shifts = torch.stack([shift_x.T.flatten(),
                              shift_y.T.flatten(),
                              shift_x.T.flatten(),
                              shift_y.T.flatten()],
                             dim=0).T

        # [K, 1, 4] & [1, A, 4] -> [K, A, 4]
        all_anchors = shifts.unsqueeze_(dim=1) + anchors.unsqueeze_(dim=0)
        # [K, A, 4] -> [K*A, 4]
        return all_anchors.reshape(-1, 4)

    def compute_featuremap_shape(self, img_shape, pyramid_level):
        """
        compute feature map's shape based on pyramid level.
        :param img_shape: 3 dimensions / [h, w, c]
        :param pyramid_level: int
        :return:
        """
        img_shape = torch.tensor(img_shape, device=self.device)
        fm_shape = (img_shape - 1) // (2 ** pyramid_level) + 1
        return fm_shape

if __name__ == "__main__":
    import pickle
    cluster_x = []
    cluster_y = []
    whs_norm = pickle.load(open("./dataset/pkl/coco_image_whs.pkl", 'rb'))
    print(skkmeans(whs_norm, 9) * 640)
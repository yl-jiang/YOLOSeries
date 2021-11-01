import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
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


if __name__ == "__main__":
    import pickle
    cluster_x = []
    cluster_y = []
    whs_norm = pickle.load(open("./dataset/pkl/coco_image_whs.pkl", 'rb'))
    print(skkmeans(whs_norm, 9) * 640)
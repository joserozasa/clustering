import math
import numpy as np
from sklearn.cluster import KMeans

import os
from datetime import datetime
import shutil
import sys

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input


def kmeans_labels(X, n_clusters):
    """Considering the case when n_clusters = 1."""
    if n_clusters == 1:
        return np.repeat(a=0, repeats=len(X))
    else:
        return KMeans(n_clusters=n_clusters, n_init='auto').fit(X).labels_


def clustering_score(X: np.ndarray, labels: np.array):
    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name]
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        if n_points_cluster > 1:
            variance = np.sum((X_cluster - centroid) ** 2) / (n_points_cluster - 1)
            loglikelihood += \
                n_points_cluster * np.log(n_points_cluster) \
                - n_points_cluster * np.log(n_points) \
                - n_points_cluster * n_dimensions * np.log(2 * math.pi * variance) \
                / 2 - (n_points_cluster - 1) / 2
    score = loglikelihood - (n_parameters / 2) * np.log(n_points)

    return score


def feature_extraction(crop_dir: str, model):
    lista_imgs = os.listdir(crop_dir)
    features_list = list()

    for img_path in lista_imgs:
        img_path = crop_dir + '/' + img_path
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features_list.append(features[0])
    return lista_imgs, features_list


def find_clusters(model, crop_dir: str, result_dir: str, min_clusters=10,
                  max_clusters=30):
    '''
    Finds  the best number of clusters, creates folders for each of them and
      writes images in.
          Parameters:
              crop_dir (str): Folder where crops are located
              result_dir (str): Folder where cluster folders will be created
              min_clusters (int): Minimum number of clusters to test
              max_clusters (int): Minimum number of clusters to test
    '''
    min_k = min_clusters
    max_k = max_clusters
    img_list, feature_list = feature_extraction(crop_dir, model)
    X = np.array(feature_list)
    all_labels = np.zeros((max_k-min_k+1, len(X)))
    scores = []
    for k in range(min_k, max_k+1):
        labels = kmeans_labels(X, n_clusters=k)
        scores.append(clustering_score(X, labels))
        all_labels[k-min_k, :] = labels
    best_k = np.nanargmax(scores) + min_k
    best_labels = dict()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    nclusters = best_k
    best_labels = all_labels[nclusters-min_k, :]
    for k in range(nclusters):
        current_dir = f"{result_dir}/{timestamp}/{k}"
        os.makedirs(current_dir)
        crop_list = [imgname for i,
                     imgname in enumerate(img_list) if best_labels[i] == k]
        for im in crop_list:
            im_path = f"{crop_dir}/{im}"
            shutil.copy(im_path, current_dir)


if __name__ == "__main__":
    image_dir = sys.argv[1]
    crop_dir = f"{image_dir}/crops"
    result_dir = f"{image_dir}/catalog"
    model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    find_clusters(model, crop_dir, result_dir)

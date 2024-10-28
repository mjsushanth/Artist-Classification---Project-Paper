import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
from scipy.spatial.distance import cdist

def test_extract_rg_info(img):
    b, g, r = cv2.split(img)
    print(img.shape)
    print(b.shape, g.shape, r.shape)

def test_extract_hog_features(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    print(h.shape)
    return h

def test_extract_sift_features(img):
    kp, des = cv2.HOGDescriptor().compute(img)
    print(des.shape)
    

def extract_sift_features(list_images, kmeans_trainer, sift):
    count = 0
    for image in list_images:
        print(count)
        count += 1
        kp, des = sift.detectAndCompute(image, None)
        kmeans_trainer.add(des)

    return kmeans_trainer

def kmean_bow(all_descriptors, num_cluster):
    bow_dict = []

    kmeans = KMeans(n_clusters = num_cluster)
    kmeans.fit(all_descriptors)

    bow_dict = kmeans.cluster_centers_

    if not os.path.isfile('bow_dictionary.pkl'):
        pickle.dump(bow_dict, open('bow_dictionary.pkl', 'wb'))

    return bow_dict

def create_feature_bow(image_descriptors, BoW, num_cluster):

    X_features = []

    for i in range(len(image_descriptors)):
        features = np.array([0] * num_cluster)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)

            argmin = np.argmin(distance, axis = 1)

            for j in argmin:
                features[j] += 1
        X_features.append(features)

    return X_features
import random
import sklearn
from sklearn.cluster import MiniBatchKMeans
import load_data
import extractor
import os
import  pickle
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import time
import gc
import joblib 
from sklearn.model_selection import GridSearchCV
from skimage import feature
import matplotlib.pyplot as plt


# train_data, train_labels, artist_id_map = load_data.label_train_image(artists_list)
# test_data, test_labels, _ = load_data.label_test_image(artists_list)
# artist_id_map = {v: k for k, v in artist_id_map.items()}

# print(len(train_data))
# load_data.store_training_data(train_data, train_labels)
# load_data.store_testing_data(test_data, test_labels)
# load_data.store_artist_id_map(artist_id_map)

# train_data = load_data.load_training_data()
# test_data, test_labels = load_data.load_testing_data()
# artist_id_map = load_data.load_artist_id_map()

# print(len(train_data))
# print(len(test_data))

# train_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))
sift = cv2.SIFT_create()

# artist_list = load_data.load_artist_names_list()
# indices_to_sample = []
# for artist in artist_list:
#     artistocc = [i for i, x in enumerate(train_labels) if x == artist_list.index(artist)]
#     indices_to_sample.extend(random.sample(artistocc, int(len(artistocc)*0.6)))

# print(len(indices_to_sample))
# print(indices_to_sample)
# indices_to_sample = pickle.load(open('indices_to_sample.pkl', 'rb'))
# # train_data = []
# kmeanstrainer = cv2.BOWKMeansTrainer(500)
# count = 0
# for i in range(1, 8):
#     print(i)
#     train_data_batch = pickle.load(open('training_data/train_data_sift_' + str(i) + '.pkl', 'rb'))
#     for ext_img in train_data_batch:
#         if count in indices_to_sample:
#             kmeanstrainer.add(ext_img)
#         count += 1
#     del train_data_batch
#     gc.collect()
# gc.collect()
# load_data.store_sift_extracted_train_data(pickle.load(open('training_data/train_data_7.pkl', 'rb')), sift, "train_data_sift_7.pkl")
# train_data = np.array( list(itertools.)  train_data)
# print(train_data.shape)
# k = 500
# batch_size = 250
# kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(train_data)

# try :
#     joblib.dump(kmeans, 'kmeans.joblib')
# except :
#     print("FAILED")

# print("COMPLETED")
# pickle.dump(kmeans, open('kmeans.pkl', 'wb')


# train_data = pickle.load(open('training_data/train_data_sift_1.pkl', 'rb'))
# for ext_img in train_data:
#     kmeanstrainer.add(ext_img)

# descriptor = kmeanstrainer.cluster()
# print(descriptor.shape)
# print(descriptor)
# pickle.dump(descriptor, open('descriptors_random_2.pkl', 'wb'))

descriptors = pickle.load(open('descriptors_random_colab.pkl', 'rb'))

# pickle.dump(kmeanstrainer, open('kmeanstrainer.pkl', 'wb'))

# pickle.dump(kmeanstrainer.cluster(), open('descriptors.pkl', 'wb'))

# for train_batch in train_data:
#     kmeanstrainer = extractor.extract_sift_features(train_batch, kmeanstrainer, sift)
#     time.sleep(30)

# descriptors = kmeanstrainer.cluster()

# print(len(data))
# # print(labels)
# # print(artist_id_map)

# print("Extracting SIFT features")

# print("Here1")
# kmeanstrainer = extractor.extract_sift_features(train_data, kmeanstrainer, sift)
# print("Here2")
# descriptors = kmeanstrainer.cluster()
# print("Here3")
# print(descriptors)
# pickle.dump(descriptors, open('descriptors.pkl', 'wb'))

# descriptors = pickle.load(open('descriptors.pkl', 'rb'))
# print(descriptors.shape)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 500)
search_params = dict(checks=400)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
bowDiction = cv2.BOWImgDescriptorExtractor(cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(descriptors)
print("bow dictionary", np.shape(descriptors))


# train_data_extract = []

# for i in range(1, 8):
#     print(i)
#     train_data_batch = pickle.load(open('training_data/train_data_' + str(i) + '.pkl', 'rb'))
#     for img in train_data_batch:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         train_data_extract.extend(bowDiction.compute(gray, sift.detect(gray)))
#     del train_data_batch
#     gc.collect()
# gc.collect()
# print(len(train_data_extract))
# pickle.dump(train_data_extract, open('train_data_extract.pkl', 'wb'))



# train_data_extract = np.array(pickle.load(open('train_data_extract.pkl', 'rb')))
# train_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))
# test_data_extract = np.array(pickle.load(open('test_dcata_extract.pkl', 'rb')))
# test_labels = pickle.load(open('testing_data/test_labels.pkl', 'rb'))

# param_grid = {'C': [0.01, 0.1, 1, 5],  
#               'gamma': ['scale'], 
#               'kernel': ['rbf']}  

# grid = GridSearchCV(sklearn.svm.SVC(), param_grid, refit = True, verbose = 3)
# grid.fit(train_data_extract, train_labels)

# print(grid.best_params_)
# print(grid.best_estimator_)

# train_data_extract = np.array(load_data.custom_extract(artists_list, bowDiction))
# print(len(train_data_extract))

# svm  = sklearn.svm.SVC(C=30, random_state=23)
# svm.fit(train_data_extract, train_labels)

# pickle.dump(svm, open('svm_model.pkl', 'wb'))
# svm = pickle.load(open('svm_model.pkl', 'rb'))
# artist_id_map = load_data.load_artist_id_map()
# artists_list = load_data.load_artist_names_list()

# # svm.train(train_data_extract, cv2.ml.ROW_SAMPLE, np.array(train_labels))

# # confusion = np.zeros((5,5))

# image_paths = load_data.get_image_paths(artists_list)
# count = 0
# correct = 0
# for image_path in image_paths:
#     # print(load_data.extract(image_path, bowDiction))
#     prediction = svm.predict(load_data.extract(image_path, bowDiction))[0]
#     print(image_path, artist_id_map[prediction])
#     # prediction = svm.predict(load_data.extract(image_path, bowDiction))[0]
#     count += 1
#     if artist_id_map[prediction] in image_path:
#         correct += 1
#     # print(svm.predict(load_data.extract(image_path, bowDiction)))

# print("Accuracy: ", correct/count)

# test_data_extract = []

# for i in range (1,3):
#     print(i)
#     test_data_batch = pickle.load(open('testing_data/test_data' + str(i) + '.pkl', 'rb'))
#     for img in test_data_batch:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         test_data_extract.extend(bowDiction.compute(gray, sift.detect(gray)))
#     del test_data_batch
#     gc.collect()
# gc.collect()
# print(len(test_data_extract))
# pickle.dump(test_data_extract, open('test_data_extract.pkl', 'wb'))
    
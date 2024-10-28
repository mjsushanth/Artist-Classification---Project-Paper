import random
import sklearn
from sklearn import svm
from sklearn.calibration import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import FeatureUnion
import load_data
import extractor
import seaborn as sns
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

#### INIT ####

# artists_list = load_data.load_artist_names_list()
# train_data, train_labels, artist_id_map = load_data.label_train_image(artists_list)
# print(len(train_labels))
# pickle.dump(train_labels, open('training_data/train_labels.pkl', 'wb'))

#### INIT ####



# train_data_batch = pickle.load(open('training_data/train_data_1.pkl', 'rb'))
# train_data_batch.extend(pickle.load(open('training_data/train_data_2.pkl', 'rb')))
# train_data_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))
# train_data_labels = np.array(train_data_labels[:2000])

# train_data_batch_hists = []
# for img in train_data_batch:
#     hist = describe(img)
#     train_data_batch_hists.append(hist)
#     print(hist)

# pickle.dump(train_data_batch_hists, open('training_data/train_data_hists.pkl', 'wb'))
# pickle.dump(train_data_labels, open('training_data/train_labels.pkl', 'wb'))

# train_data_batch = pickle.load(open('training_data/train_data_hists.pkl', 'rb'))
# train_data_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))

# X_train, X_test, y_train, y_test = train_test_split(train_data_batch, train_data_labels, test_size=0.2, random_state=42)
# X_train =  np.array(X_train)
# y_train = np.array(y_train)

# print(X_train.shape)
# print(y_train.shape)

# model = svm.SVC(C=100000, random_state=15)

# # model = LinearSVC(C=1000000, random_state=15,dual=False, max_iter=10000)
# model.fit(X_train, y_train)

# print(model.score(X_test, y_test))




# def describe(image, eps = 1e-7):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lbp = feature.local_binary_pattern(gray, 24, 8, method="uniform")
#     (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))


#     hist = hist.astype("float")
#     hist /= (hist.sum() + eps)

#     return hist

# train_data_batch_hists = []
# for i in range(1,8):
#     print(i)
#     train_data_batch = pickle.load(open('training_data/train_data_' + str(i) + '.pkl', 'rb'))
#     for img in train_data_batch:
#         hist = describe(img)
#         train_data_batch_hists.append(hist)
#     del train_data_batch
#     gc.collect()
# gc.collect()

# img = cv2.imread('project_data/images_resized/Amedeo_Modigliani/train/Amedeo_Modigliani_3.jpg', 1)
# pixels = np.float32(img.reshape(-1, 3))
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# k = 10
# # _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# # _, counts = np.unique(labels, return_counts=True)

# kmeans = MiniBatchKMeans(n_clusters=k, batch_size=250).fit(pixels)
# palette = kmeans.cluster_centers_

# print(palette.flatten())
# print(palette)

# train_data_color_test = []


# for i in range(1,8):
#     print(i)
#     train_data_batch = pickle.load(open('training_data/train_data_' + str(i) + '.pkl', 'rb'))
#     count = 0
#     for img in train_data_batch:
#         print(count)
#         count += 1
#         pixels = np.float32(img.reshape(-1, 3))
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#         k = 10
#         palette = MiniBatchKMeans(n_clusters=k, batch_size=250).fit(pixels).cluster_centers_
#         # _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#         # _, counts = np.unique(labels, return_counts=True)

#         # dominant = palette[np.argmin(counts)]
#         train_data_color_test.append(palette.flatten())
#     del train_data_batch
#     gc.collect()

# pickle.dump(train_data_color_test, open('training_data/train_data_color.pkl', 'wb'))

# train_data_color = pickle.load(open('training_data/train_data_color.pkl', 'rb'))
train_data_hist = np.array(pickle.load(open('training_data/train_data_hists.pkl', 'rb'))) 
train_data_bow = np.array(pickle.load(open('train_data_extract.pkl', 'rb')))
train_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))
train_genres = np.array(pickle.load(open('training_data/train_genres.pkl', 'rb')))
# train_nationalities = np.array(pickle.load(open('training_data/train_nationalities.pkl', 'rb')))
print(train_genres.shape)


# scaler_bow = sklearn.preprocessing.StandardScaler().fit(train_data_bow)
# train_data_bow = scaler_bow.transform(train_data_bow)

# scaler_color = sklearn.preprocessing.StandardScaler().fit(train_data_color)
# train_data_color = scaler_color.transform(train_data_color)
# train_data = np.concatenate((train_data_color, train_data_hist), axis=1)
# train_data = np.concatenate((train_data_color, train_data_bow), axis=1)
# print(train_data[0].shape)

# union = FeatureUnion([("color", train_data_color), ("hist", train_data_bow)])

# scaler = sklearn.preprocessing.StandardScaler().fit(train_data)
# train_data = scaler.transform(train_data)

# train_rpalette=[]
# train_gpalette=[]
# train_bpalette=[]

# for pal in train_data_color:    
#     rindex=0
#     gindex=1
#     bindex=2    

#     rs=[]
#     bs=[]
#     gs=[]
#     while gindex < len(pal):
#         rs.append(pal[rindex])
#         gs.append(pal[gindex])
#         bs.append(pal[bindex])
#         rindex+=3
#         gindex+=3
#         bindex+=3
#     train_rpalette.append(rs)
#     train_gpalette.append(gs)
#     train_bpalette.append(bs)

# train_rpalette=np.array(train_rpalette)
# train_gpalette=np.array(train_gpalette) 
# train_bpalette=np.array(train_bpalette) 

# scaler_rpalette = sklearn.preprocessing.StandardScaler().fit(train_rpalette)
# train_rpalette = scaler_rpalette.transform(train_rpalette)

# scaler_gpalette = sklearn.preprocessing.StandardScaler().fit(train_gpalette)
# train_gpalette = scaler_gpalette.transform(train_gpalette)

# scaler_bpalette = sklearn.preprocessing.StandardScaler().fit(train_bpalette)
# train_bpalette = scaler_bpalette.transform(train_bpalette)


# scaler = sklearn.preprocessing.StandardScaler().fit(train_data_color)
# train_data_color = scaler.transform(train_data_color)

# train_data = np.concatenate((train_data, train_data_hist), axis=1)
tuned_parameters={"kernel":['rbf'],
                 'C':[1000,1010,1020,1030,1040]}

train_data = np.concatenate((train_data_bow, train_genres), axis=1)
# train_data = np.concatenate((train_data, train_nationalities), axis=1)
train_data = np.concatenate((train_data, train_data_hist), axis=1)

# train_data = np.concatenate((train_data, train_rpalette), axis=1)
# train_data = np.concatenate((train_data, train_gpalette), axis=1)
# train_data = np.concatenate((train_data, train_bpalette), axis=1)

# grid = GridSearchCV(svm.SVC(), tuned_parameters,refit = True, verbose = 3)
# grid.fit(train_data, np.array(train_labels))
# print(grid.best_params_)
# print(grid.best_estimator_)

testing_data_bow = np.array(pickle.load(open('test_data_extract.pkl', 'rb')))
testing_data_genres = np.array(pickle.load(open('testing_data/test_genres.pkl', 'rb')))
testing_data_hists = np.array(pickle.load(open('testing_data/test_data_hists.pkl', 'rb')))
testing_labels = np.array(pickle.load(open('testing_data/test_labels.pkl', 'rb')))

# print(testing_data_bow[1].shape)

testing_data = np.concatenate((testing_data_bow, testing_data_genres), axis=1)
testing_data = np.concatenate((testing_data, testing_data_hists), axis=1)
# print(testing_data.shape)
# print(testing_labels.shape)

# model = svm.SVC(C=1020, random_state=15)
# model.fit(train_data, np.array(train_labels))
model = pickle.load(open('fml.pkl', 'rb'))

pickle.dump(model, open('fml.pkl', 'wb'))

print(model.score(testing_data, testing_labels))

confusion_matrix = sklearn.metrics.confusion_matrix(testing_labels, model.predict(testing_data))
artists_list = load_data.load_artist_names_list()
print(confusion_matrix)
heatmap = sns.heatmap(confusion_matrix, annot=True, xticklabels=artists_list, yticklabels=artists_list, cmap='Blues', annot_kws={"fontsize":8}, fmt='d')
plt.xticks( rotation='vertical', fontsize=5)
plt.yticks( rotation='horizontal', fontsize=5)
plt.figure(figsize=(50,50))
fig = heatmap.get_figure()
fig.savefig('confusion_matrix.png')
# plt.show()
artist_map = load_data.load_artist_id_map()

metric_labels = [artist_map[label] for label in testing_labels]

print(sklearn.metrics.classification_report(testing_labels, model.predict(testing_data)))
# print(sklearn.metrics.roc_auc_score(testing_labels, model.predict(testing_data), average='weighted', multi_class='ovr'))
# pickle.dump(train_data_batch_hists, open('training_data/train_data_hists'+ '.pkl', 'wb'))


# test_data_hist = []

# for i in range(1,3):
#     print(i)
#     test_data_batch = pickle.load(open('testing_data/test_data' + str(i) + '.pkl', 'rb'))
#     for img in test_data_batch:
#         hist = describe(img)
#         test_data_hist.append(hist)
#     del test_data_batch
#     gc.collect()
# gc.collect()

# descriptors = pickle.load(open('descriptors_random_2.pkl', 'rb'))
# train_data_extract = np.array(pickle.load(open('train_data_extract.pkl', 'rb')))
# train_data_hist = pickle.load(open('training_data/train_data_hists.pkl', 'rb'))
# train_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))

# train_data_extract = np.array(train_data_extract)
# train_data_hist = np.array(train_data_hist)

# train_data = np.concatenate((train_data_extract, train_data_hist), axis=1)

# train_data = [[np.concatenate(train_data_extract[i], train_data_hist[i])] for i in range(len(train_data_hist))]
# print(train_data[0])
# print(train_data[0][0].shape)
# train_data = np.array(train_data)

# model = svm.SVC(C=100000, random_state=15)
# model.fit(train_data, np.array(train_labels))


# grid = GridSearchCV(svm.SVC(), {'C': [1, 10, 100, 1000, 10000, 100000]},refit = True, verbose = 3)

# grid.fit(train_data, train_labels)

# print(grid.best_params_)
# print(grid.best_estimator_)
# pickle.dump(model, open('fml.pkl', 'wb'))

# pickle.dump(test_data_hist, open('testing_data/test_data_hists.pkl', 'wb'))

# model = pickle.load(open('fml.pkl', 'rb'))

# test_data_hist = pickle.load(open('testing_data/test_data_hists.pkl', 'rb'))

# test_data_hist = np.array(test_data_hist)
# test_data_extract = np.array(pickle.load(open('test_data_extract.pkl', 'rb')))
# test_data = np.concatenate((test_data_extract, test_data_hist), axis=1)
# test_labels = pickle.load(open('testing_data/test_labels.pkl', 'rb'))

# print(model.score(test_data, test_labels))

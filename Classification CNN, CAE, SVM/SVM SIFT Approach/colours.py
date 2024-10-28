import gc
import pickle
import cv2 
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76
from sklearn.cluster import MiniBatchKMeans

rgb = cv2.imread('project_data/images_resized/Amedeo_Modigliani/train/Amedeo_Modigliani_3.jpg', 1)

lab_color = rgb2lab(rgb)

print(lab_color)

train_data_color = []

for i in range(1,8):
    train_data_batch = pickle.load(open('training_data/train_data_' + str(i) + '.pkl', 'rb'))
    for img in train_data_batch:
        lab_color = rgb2lab(img)
        lab_pixels = np.float32(lab_color.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 10
        palette = MiniBatchKMeans(n_clusters=k, batch_size=250).fit(lab_pixels).cluster_centers_
        train_data_color.append(palette.flatten())
    del train_data_batch
    gc.collect()

pickle.dump(train_data_color, open('training_data/train_data_color.pkl', 'wb'))

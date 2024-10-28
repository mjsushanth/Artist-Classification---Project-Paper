import pickle
import numpy as np
import matplotlib.pyplot as plt
training_data = np.array(pickle.load(open('train_data_extract.pkl', 'rb')))
training_labels = pickle.load(open('train_labels.pkl', 'rb'))

artist_id_map = pickle.load(open('artist_id_map.pkl', 'rb'))


print(artist_id_map[training_labels[6857]])
plt.hist(training_data[6857])
plt.savefig('inpres/jackson_pollock_hist_3.png')
plt.show()


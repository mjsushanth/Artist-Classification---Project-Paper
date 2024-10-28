import os
import pandas as pd
from unidecode import unidecode
import random
import pickle


root_dir = "project_data"
images_root_dir = os.path.join(root_dir, "images_resized")


info = pd.read_csv(os.path.join(root_dir, "artists.csv"))
names = info['name']
artist_names_list = [ unidecode(name.replace(" ", "_")) for name in  names.tolist()] 

print(artist_names_list)

pickle.dump(artist_names_list, open('artist_names_list.pkl', 'wb'))

# for image in os.listdir(images_root_dir):
#     for artist in artist_names_list:
#         if artist in image:
#             artist_path = os.path.join(images_root_dir, artist)
#             print(image, artist_path)
#             if not os.path.exists(artist_path):
#                 os.makedirs(artist_path)
#             os.rename(os.path.join(images_root_dir, image), os.path.join(artist_path, image))
#             break


# for artist in artist_names_list:
#     artist_path = os.path.join(images_root_dir, artist)
#     if not os.path.exists(artist_path):
#         os.makedirs(artist_path)
    
#     for image in os.listdir(images_root_dir):
#         print(image, artist_path)
#         if artist in image:
#             os.rename(os.path.join(images_root_dir, image), os.path.join(artist_path, image))
    

# for artist in artist_names_list:
#     artist_path = os.path.join(images_root_dir, artist)
#     artist_path_train = os.path.join(artist_path, "train")
#     artist_path_test = os.path.join(artist_path, "test")

#     if not os.path.exists(artist_path_train):
#         os.makedirs(artist_path_train)
#     if not os.path.exists(artist_path_test):
#         os.makedirs(artist_path_test)

#     for image in os.listdir(artist_path):
#         if not os.path.isdir(os.path.join(artist_path, image)):
#             randnum = random.randint(0, 100)
#             if randnum < 80: 
#                 os.rename(os.path.join(artist_path, image), os.path.join(artist_path_train, image))
#             else:
#                 os.rename(os.path.join(artist_path, image), os.path.join(artist_path_test, image))
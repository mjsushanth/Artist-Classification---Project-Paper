import os
import cv2
import pickle
import random
import time

images_root = "project_data/images_resized"


def load_artist_names_list():
    artists_list = pickle.load(open('artist_names_list.pkl', 'rb'))
    return artists_list

def label_train_image(artists_list):
    data = []
    labels = []
    artist_id_map = {}
    id = 0
    
    for artist in artists_list:
        artist_id_map[artist] = id
        id += 1
        artist_path = os.path.join(images_root, artist, "train")
        for image in os.listdir(artist_path):
            image_path = os.path.join(artist_path, image)

            if os.path.isfile(image_path):
                img = cv2.imread(image_path, 1)
                data.append(img)
                labels.append(artist_id_map[artist])
    
    return data, labels, artist_id_map

def label_test_image(artists_list):
    data = []
    labels = []
    artist_id_map = {}
    id = 0
    
    for artist in artists_list:
        artist_id_map[artist] = id
        id += 1
        artist_path = os.path.join(images_root, artist, "test")
        for image in os.listdir(artist_path):
            image_path = os.path.join(artist_path, image)

            if os.path.isfile(image_path):
                img = cv2.imread(image_path, 1)
                data.append(img)
                labels.append(artist_id_map[artist])
    
    return data, labels, artist_id_map


def extract(image_path, bowdict):
    sift = cv2.SIFT_create()
    img = cv2.imread(image_path, 1)
    return bowdict.compute(img, sift.detect(img))

def custom_extract(artist_list, boWdict):
    train_list = [ ]
    sift = cv2.SIFT_create()
    for artist in artist_list:
        artist_path = os.path.join(images_root, artist, "train")
        for image in os.listdir(artist_path):
            image_path = os.path.join(artist_path, image)

            if os.path.isfile(image_path):
                img = cv2.imread(image_path, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                train_list.extend(boWdict.compute(gray, sift.detect(gray)))
    return train_list

def custom_extract_test(artist_list, BoWdict):
    test_list = [ ]
    sift = cv2.SIFT_create()
    for artist in artist_list:
        artist_path = os.path.join(images_root, artist, "test")
        for image in os.listdir(artist_path):
            image_path = os.path.join(artist_path, image)

            if os.path.isfile(image_path):
                img = cv2.imread(image_path, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                test_list.extend(BoWdict.compute(gray, sift.detect(gray)))
    return test_list


def get_image_paths(artist_list):
    image_paths = []
    for artist in artist_list:
        artist_path = os.path.join(images_root, artist, "test")
        for image in os.listdir(artist_path):
            image_path = os.path.join(artist_path, image)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
    return image_paths


def store_training_data(train_data, train_labels):    
    pickle.dump(train_data[:1000], open('training_data/train_data_1.pkl', 'wb'))
    time.sleep(30)
    print("First batch stored")
    pickle.dump(train_data[1000:2000], open('training_data/train_data_2.pkl', 'wb'))
    time.sleep(30)
    print("Second batch stored")
    pickle.dump(train_data[2000:3000], open('training_data/train_data_3.pkl', 'wb'))
    time.sleep(30)
    print("Third batch stored")
    pickle.dump(train_data[3000:4000], open('training_data/train_data_4.pkl', 'wb'))
    time.sleep(30)
    print("Fourth batch stored")
    pickle.dump(train_data[4000:5000], open('training_data/train_data_5.pkl', 'wb'))
    time.sleep(30)
    print("Fifth batch stored")
    pickle.dump(train_data[5000:6000], open('training_data/train_data_6.pkl', 'wb'))
    time.sleep(30)
    print("Sixth batch stored")
    pickle.dump(train_data[6000:7000], open('training_data/train_data_7.pkl', 'wb'))
    time.sleep(30)
    print("Seventh batch stored")
    pickle.dump(train_labels, open('training_data/train_labels.pkl', 'wb'))
    print("Labels stored")

def store_testing_data(test_data, test_labels):
    pickle.dump(test_data[:1000], open('testing_data/test_data1.pkl', 'wb'))
    time.sleep(30)
    print("First batch stored")
    pickle.dump(test_data[1000:2000], open('testing_data/test_data2.pkl', 'wb'))
    time.sleep(30)
    print("Second batch stored")
    pickle.dump(test_data[2000:3000], open('testing_data/test_data3.pkl', 'wb'))
    time.sleep(30000)
    print("Third batch stored")
    pickle.dump(test_labels, open('testing_data/test_labels.pkl', 'wb'))
    print("Labels stored")

def store_artist_id_map(artist_id_map):
    pickle.dump(artist_id_map, open('artist_id_map.pkl', 'wb'))    

def load_artist_id_map():
    artist_id_map = pickle.load(open('artist_id_map.pkl', 'rb'))
    return artist_id_map

def store_sift_extracted_train_data(train_data, sift, name):
    sift_data = []
    for image in train_data:
        kp, des = sift.detectAndCompute(image, None)
        sift_data.append(des)
    pickle.dump(sift_data, open("training_data/" + name, 'wb'))
    

def load_training_data():
    train_data = []
    train_data.extend(pickle.load(open('training_data/train_data_sift_1.pkl', 'rb')))
    print(train_data[0].shape)
    print(train_data[1].shape)
    print(train_data[273].shape)
    time.sleep(10)
    train_data.extend(pickle.load(open('training_data/train_data_sift_2.pkl', 'rb')))
    time.sleep(10)
    train_data.extend(pickle.load(open('training_data/train_data_sift_3.pkl', 'rb')))
    time.sleep(10)
    train_data.extend(pickle.load(open('training_data/train_data_sift_4.pkl', 'rb')))
    time.sleep(10)
    train_data.extend(pickle.load(open('training_data/train_data_sift_5.pkl', 'rb')))
    time.sleep(10)
    train_data.extend(pickle.load(open('training_data/train_data_sift_6.pkl', 'rb')))
    time.sleep(10)
    train_data.extend(pickle.load(open('training_data/train_data_sift_7.pkl', 'rb')))
    time.sleep(10)
    # train_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))
    # time.sleep(10)
    return train_data #, train_labels

# def load_testing_data():
#     test_data = []
#     test_data.append(pickle.load(open('testing_data/test_data1.pkl', 'rb')))
#     time.sleep(10)
#     test_data.append(pickle.load(open('testing_data/test_data2.pkl', 'rb')))
#     time.sleep(10)
#     test_labels = pickle.load(open('testing_data/test_labels.pkl', 'rb'))
#     return test_data, test_labels
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
from src.constants import Monet_jpg, Photo_jpg

def load_images(path, size=(256, 256)):
    """
    Load images from a directory, convert to arrays, and scale.
    """
    data_list = []
    file_list = os.listdir(path)
    for file in file_list:
        img = load_img(os.path.join(path, file), target_size=size)
        img = img_to_array(img)
        img = img / 127.5 - 1  # Scale to [-1, 1]
        data_list.append(img)
    return np.asarray(data_list)

def get_image_statistics(path):
    """
    Compute and return image dimensions statistics.
    """
    Dim1, Dim2, CS = [], [], []
    for ele in os.listdir(path):
        img = load_img(os.path.join(path, ele))
        img_array = img_to_array(img)
        D1, D2, C = img_array.shape
        Dim1.append(D1)
        Dim2.append(D2)
        CS.append(C)
    df = pd.DataFrame({'D1': Dim1, 'D2': Dim2, 'C': CS})
    return df.describe()

def display_sample_images(Monet_arr, Photo_arr):
    """
    Display sample Monet painting and Photo.
    """
    # Sample Monet painting
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow((Monet_arr[0] + 1) / 2.0)
    plt.title('Sample Monet Painting')
    plt.axis('off')
    
    # Sample Photo
    plt.subplot(1,2,2)
    plt.imshow((Photo_arr[0] + 1) / 2.0)
    plt.title('Sample Photo')
    plt.axis('off')
    
    plt.show() 
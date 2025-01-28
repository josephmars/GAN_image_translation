import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
from src.utils import pred_save
from src.constants import IMG_SHAPE

def perform_inference(g_PHtoM, Photo_arr, output_dir='/kaggle/working/images/'):
    """
    Generate Monet-style images from photos and save them.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(Photo_arr.shape[0]):
        img_in = Photo_arr[i][np.newaxis, :]
        pred_save(img_in, g_PHtoM, output_dir, i)
    
    n = len(os.listdir(output_dir))
    print(f'{n} images generated!')

def display_random_generated_image(path):
    """
    Display a randomly selected generated image.
    """
    n = len(os.listdir(path))
    i = np.random.randint(0, n)
    sample = os.path.join(path, os.listdir(path)[i])
    img = plt.imread(sample)
    print(img.shape)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def load_trained_model(model_path):
    """
    Load a trained generator model.
    """
    from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
    cust = {'InstanceNormalization': InstanceNormalization}
    model = load_model(model_path, cust)
    return model 
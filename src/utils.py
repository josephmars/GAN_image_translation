import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import PIL
from src.constants import IMG_SHAPE

def generate_real_samples(dataset, n_samples, patch_shape):
    """
    Select a batch of random real samples.
    """
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

def generate_fake_samples(g_model, dataset, patch_shape):
    """
    Generate a batch of fake samples.
    """
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def fake_pool(pool, images, max_size=100):
    """
    Create a pool of fake images.
    """
    selected = []
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5:
            selected.append(image)
        else:
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)

def save_models(step, g_MtoPH, g_PHtoM, save_dir='src/models/'):
    """
    Save generator models to local files.
    """
    os.makedirs(save_dir, exist_ok=True)
    name1 = os.path.join(save_dir, f'g_MtoPH_{step+1:06d}.h5')
    g_MtoPH.save(name1)

    name2 = os.path.join(save_dir, f'g_PHtoM_{step+1:06d}.h5')
    g_PHtoM.save(name2)
    print(f"Done saving step {step+1}")

def pred_save(img_in, G_model, path, i):
    """
    Generate Monet-style image and save to the specified path.
    """
    prediction = G_model.predict(img_in)
    prediction = prediction[0]
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)  # Rescale to [0, 255]
    im = PIL.Image.fromarray(prediction)
    im.save(f'{path}{str(i)}.jpg')

def plot_generated_images(real, fake, title_real='Real Image', title_fake='Generated Image'):
    """
    Display real and generated images side by side.
    """
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow((real + 1) / 2.0)
    plt.title(title_real)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow((fake + 1) / 2.0)
    plt.title(title_fake)
    plt.axis('off')
    
    plt.show() 
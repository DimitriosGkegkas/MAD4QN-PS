import os
import numpy as np
import matplotlib.pyplot as plt

def debug_observation(observation_array, save_folder='./debug'):
    # Ensure observation_array is a numpy array
    observation_array = np.array(observation_array)
    
    # Check the shape for debugging
    print(f"Observation array shape: {observation_array.shape}")
    
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    if observation_array.ndim == 3:  # (num_images, height, width)
        num_images = observation_array.shape[0]
        for i in range(num_images):
            # Save each image
            image_path = os.path.join(save_folder, f'image_{i+1}.png')
            plt.imsave(image_path, observation_array[i], cmap='gray')  # Assuming grayscale
            print(f"Saved image {i+1} to {image_path}")
    else:
        print("Observation array is not 3D (num_images, height, width).")


def debug_save_any_img(img, save_folder='./debug'):
    # Ensure img is a numpy array
    img = np.array(img)
    shape = img.shape
    
    # Check the shape for debugging
    print(f"Image shape: {shape}")
    
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # maybe the image is (height, width, channels) or (channels, height, width) or (height, width) or (batch, height, width) with batch >3
    if img.ndim == 3 and shape[0] == 3:
        img = np.reshape(img, (shape[1], shape[2], shape[0]))
        plt.imsave(os.path.join(save_folder, 'image.png'), img)
    elif img.ndim == 3 and shape[2] == 3:
        plt.imsave(os.path.join(save_folder, 'image.png'), img)
    elif img.ndim == 3 and shape[0] > 3:
        # group of images of 3 channels
        for i in range(shape[0]//3):
            plt.imsave(os.path.join(save_folder, f'image_{i+1}.png'), np.reshape(img[i*3:i*3+3], (shape[1], shape[2], 3)))
    elif img.ndim == 2:
        plt.imsave(os.path.join(save_folder, 'image.png'), img, cmap='gray')

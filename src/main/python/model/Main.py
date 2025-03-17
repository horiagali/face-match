import os
import matplotlib.pyplot as plt
from PIL import Image
from Data_loader import load_and_preprocess_test_data

import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2

def visualize_images(images, num_images=5):
    """
    Visualize the first few images from the dataset.
    
    Args:
        images (np.ndarray): Array of preprocessed images.
        num_images (int): Number of images to visualize.
    """
    for i in range(num_images):
        plt.figure(figsize=(2, 2))
        plt.imshow(images[i].reshape(48, 48), cmap='gray')  # Reshape to (48, 48) for grayscale display
        plt.title(f'Image {i + 1}')
        plt.axis('off')
        plt.show()

def visualize_feature_maps(feature_maps, num_maps=5):
    """
    Visualize the feature maps after applying the CNN layers.
    
    Args:
        feature_maps (np.ndarray): The feature maps after CNN layers.
        num_maps (int): Number of feature maps to visualize.
    """
    # If the feature maps have only one channel, reshape to 3D for uniformity
    if len(feature_maps.shape) == 2:  # If the output is 2D
        feature_maps = feature_maps[:, :, np.newaxis]  # Add a singleton dimension for channels

    num_feature_maps = feature_maps.shape[-1]  # Number of channels (filters)
    
    for i in range(min(num_maps, num_feature_maps)):
        plt.figure(figsize=(2, 2))
        plt.imshow(feature_maps[:, :, i], cmap='gray')  # Display each feature map as grayscale
        plt.title(f'Feature Map {i + 1}')
        plt.axis('off')
        plt.show()


def convolution_2d(image, filter, step):
    """
    Perform a 2D convolution operation on an image with a given filter.
    """
    k_size = filter.shape[0]
    width_out = int((image.shape[0] - k_size) / step + 1)
    height_out = int((image.shape[1] - k_size) / step + 1)
    output_image = np.zeros((width_out, height_out))
    
    for i in range(0, width_out, step):
        for j in range(0, height_out, step):
            patch_from_image = image[i:i+k_size, j:j+k_size]
            output_image[i, j] = np.sum(patch_from_image * filter)
    
    return output_image

def cnn_layer(image_volume, filter, step=1):
    """
    Apply a convolutional layer on the input image volume with a given filter.
    """
    k_size = filter.shape[0]
    width_out = int((image_volume.shape[0] - k_size) / step + 1)
    height_out = int((image_volume.shape[1] - k_size) / step + 1)
    
    # Only one depth dimension since it's grayscale
    feature_map = np.zeros((width_out, height_out))
    
    feature_map = convolution_2d(image_volume, filter, step)
    
    return feature_map


def relu_layer(maps):
    """
    Apply ReLU activation function.
    """
    return np.maximum(0, maps)

def pooling_layer(maps, size=2, step=2):
    """
    Apply max pooling.
    """
    width_out = int((maps.shape[0] - size) / step + 1)
    height_out = int((maps.shape[1] - size) / step + 1)
    pooling_image = np.zeros((width_out, height_out))
    
    for i in range(0, width_out, step):
        for j in range(0, height_out, step):
            patch_from_image = maps[i:i+size, j:j+size]
            pooling_image[i, j] = np.max(patch_from_image)
    
    return pooling_image

def fully_connected_layer(feature_map, output_size=7):
    """
    Fully connected layer for multi-class classification (7 emotions).
    """
    flattened = feature_map.flatten()
    weights = np.random.randn(flattened.shape[0], output_size)
    output = np.dot(flattened, weights)
    return output

def softmax(logits):
    """
    Apply softmax to logits to get probabilities.
    """
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    return exp_logits / np.sum(exp_logits)


def apply_cnn_blocks(image, filters_list, step=1, pool_size=2, pool_step=2):
    """
    Applies multiple convolutional layers sequentially with ReLU activation and pooling.

    Args:
        image (np.ndarray): The input image.
        filters_list (list of np.ndarray): List of filters for each convolutional layer.
        step (int): Stride for convolution.
        pool_size (int): Pooling window size.
        pool_step (int): Stride for pooling.

    Returns:
        np.ndarray: The output feature map after the CNN layers.
    """
    output = image  

    for filter in filters_list:
        output = cnn_layer(output, filter, step=step)  # Convolution
        output = relu_layer(output)                     # ReLU activation
        output = pooling_layer(output, size=pool_size, step=pool_step)  # Pooling

    return output

def main():

    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    num_images = 10  # how many images i take

    # Hyperparameters
    K_number = 2    # Number of filters per layer
    K_size = 3      # Filter size
    step = 1        # Convolution stride
    target_size = (48, 48)  # Image size
    num_layers = 3  # Number of convolutional blocks

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, '..', '..', 'resources', 'faces')
    
    test_csv_path = os.path.join(data_folder, 'test.csv')

    if not os.path.exists(test_csv_path):
        print(f"The file {test_csv_path} does not exist.")
        return

    # Load and preprocess data
    images = load_and_preprocess_test_data(test_csv_path, target_size)

    images = images[:num_images]

    # Initialize random filters for each layer
    filters_list = [np.random.randn(K_size, K_size) for _ in range(num_layers)]  # Only 2D filters for grayscale

    feature_vectors = []

    for idx, img in enumerate(tqdm(images, desc="Processing Images")):
        # Create a figure with 1 row and 2 columns
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # Two plots: image and probability bar chart

        # Plot the image
        ax[0].imshow(img.reshape(48, 48), cmap='gray')  # Reshape to (48, 48) for grayscale display
        ax[0].set_title(f'Image {idx + 1}')
        ax[0].axis('off')  # Hide axes

        # Apply CNN blocks
        feature_maps = apply_cnn_blocks(img, filters_list, step=step)

        # Fully connected layer output and softmax probabilities
        fc_output = fully_connected_layer(feature_maps)  # Fully connected layer
        probabilities = softmax(fc_output)  # Apply softmax to get probabilities

        feature_vectors.append(probabilities)  # Append probabilities

        # Plot the probabilities as a bar chart
        ax[1].bar(emotions, probabilities * 100, color='blue')
        ax[1].set_title('Predicted Probabilities')
        ax[1].set_ylabel('Probability (%)')
        ax[1].set_ylim([0, 100])  # Set the y-axis limit to 0-100% for better scaling

        # Display the image with the bar chart
        plt.tight_layout()
        plt.show()

        # Print the probabilities below the image in the console as well
        print(f"Image {idx + 1} probabilities:")
        for emotion, prob in zip(emotions, probabilities):
            print(f"  {emotion}: {prob * 100:.2f}%")
        print()  

    # Convert feature_vectors to a numpy array
    feature_vectors = np.array(feature_vectors)

if __name__ == "__main__":
    main()


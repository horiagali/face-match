import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_and_preprocess_data(image_folder, target_size, pad):
    """
    Load and preprocess images from the given folder.
    Args:
        image_folder (str): Path to the folder containing the images.
        target_size (tuple): The target size to which all images will be resized (default: (224, 224)).
    Returns:
        np.ndarray: Array of preprocessed images.
    """
    processed_images = []

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        
        img = load_image(image_path)
        
        img = preprocess_image(img, target_size, pad)
        
        if img is not None:
            processed_images.append(img)
    
    return np.array(processed_images)

def load_image(image_path):
    """
    Load an image from the specified path.
    """
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(img, target_size , pad_width):
    """
    Preprocess the image by resizing, normalizing, and adding padding.
    """
    if img is not None:
        img = img.resize(target_size)
        img = img.convert('RGB')  # Ensures the image is in RGB format
        
        # Add padding of pad_width (default 1) around the image
        img_padded = np.pad(np.array(img), ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
        
        return img_padded
    else:
        return None
import numpy as np


def convolution_2d(image, filter, pad, step):
    """
    Perform a 2D convolution operation on an image with a given filter.
    
    Args:
        image (np.ndarray): The input image to be convolved.
        filter (np.ndarray): The filter (kernel) to apply.
        pad (int): The padding applied to the image.
        step (int): The step size (stride) for the convolution.
    
    Returns:
        np.ndarray: The resulting image after convolution.
    """
    k_size = filter.shape[0]

    width_out = int((image.shape[0] - k_size + 2 * pad) / step + 1)
    height_out = int((image.shape[1] - k_size + 2 * pad) / step + 1)

    output_image = np.zeros((width_out - 2 * pad, height_out - 2 * pad))

    for i in range(image.shape[0] - k_size + 1):
        for j in range(image.shape[1] - k_size + 1):
            patch_from_image = image[i:i+k_size, j:j+k_size]
            output_image[i, j] = np.sum(patch_from_image * filter)

    return output_image


def cnn_layer(image_volume, filter, pad=1, step=1):
    """
    Apply a convolutional layer on the input image volume with a given filter set.
    
    Args:
        image_volume (np.ndarray): The input image volume (height x width x depth).
        filter (np.ndarray): The filter set (number_of_filters x filter_height x filter_width x input_depth).
        pad (int): The padding applied to the image (default is 1).
        step (int): The step size (stride) for the convolution (default is 1).
    
    Returns:
        np.ndarray: The feature maps after applying the convolutional layer.
    """
    image = np.zeros((image_volume.shape[0] + 2 * pad, image_volume.shape[1] + 2 * pad, image_volume.shape[2]))

    for p in range(image_volume.shape[2]):
        image[:, :, p] = np.pad(image_volume[:, :, p], (pad, pad), mode='constant', constant_values=0)

    k_size = filter.shape[1]
    depth_out = filter.shape[0]
    width_out = int((image_volume.shape[0] - k_size + 2 * pad) / step + 1)
    height_out = int((image_volume.shape[1] - k_size + 2 * pad) / step + 1)

    feature_maps = np.zeros((width_out, height_out, depth_out))

    n_filters = filter.shape[0]

    for i in range(n_filters):
        convolved_image = np.zeros((width_out, height_out))

        for j in range(image.shape[-1]):
            convolved_image += convolution_2d(image[:, :, j], filter[i, :, :, j], pad, step)
        
        feature_maps[:, :, i] = convolved_image

    return feature_maps


def image_pixels_255(maps):

    """
    Replaces values in an image over 255 with 255
    """
    r = np.zeros(maps.shape)

    for c in range(r.shape[2]):
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if maps[i, j, c] <= 255:
                    r[i, j, c] = maps[i, j, c]
                else:
                    r[i, j, c] = 255
    return r

def relu_layer(maps):
    """
  
    Args:
        maps (np.ndarray): The input feature maps, which can be the output of a convolutional layer.
    
    Returns:
        np.ndarray: The feature maps after applying the ReLU activation function, 
                    where all negative values are replaced with zeros.
    """
    r = np.zeros_like(maps)
    
    result = np.where(maps > r, maps, r)
    
    return result

def pooling_layer(maps, size=2, step=2):
    """
    Apply the max pooling operation to the input feature maps.

    Args:
        maps (np.ndarray): The input feature maps (height x width x depth).
        size (int): The size of the pooling filter (default is 2x2).
        step (int): The stride (step size) for the pooling operation (default is 2).

    Returns:
        np.ndarray: The pooled feature maps with reduced spatial dimensions.
    """
    width_out = int((maps.shape[0] - size) / step + 1)
    height_out = int((maps.shape[1] - size) / step + 1)

    pooling_image = np.zeros((width_out, height_out, maps.shape[2]))

    for c in range(maps.shape[2]):
        ii = 0
        for i in range(0, maps.shape[0] - size + 1, step):
            jj = 0
            for j in range(0, maps.shape[1] - size + 1, step):
                patch_from_image = maps[i:i+size, j:j+size, c]
                pooling_image[ii, jj, c] = np.max(patch_from_image)
                jj += 1
            ii += 1

    return pooling_image

def fully_connected_layer(feature_map, output_size=10):
    """
    A simple fully connected layer that takes in the feature map and outputs a vector of size `output_size`.
    
    Args:
        feature_map (np.ndarray): Flattened feature map (a 1D array of features).
        output_size (int): The number of output classes (default is 10).
    
    Returns:
        np.ndarray: The final output vector of size `output_size`.
    """
    flattened = feature_map.flatten()
    
    #TODO train weights
    weights = np.random.randn(flattened.shape[0], output_size)
    
    # Output vector
    output = np.dot(flattened, weights)
    
    return output

def main():
    # Hyperparameters:
    K_number = 2    # number of filters (kernels) denoted as K_number
    K_size = 3      # size of filters (spatial dimension) denoted as K_size
    step = 1        # step for sliding (also known as stride) denoted as Step
    pad = 1         # processing edges by zero-padding parameter denoted as Pad
    target_size = (224, 224)  # image size

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
    image_folder = os.path.join(project_root, 'src', 'main', 'resources', 'faces')
    
    if not os.path.exists(image_folder):
        print(f"The directory {image_folder} does not exist.")
        return
    
    # Load and preprocess data
    images = load_and_preprocess_data(image_folder, target_size, pad)
    
    # TODO train filters
    filters = np.random.randn(K_number, K_size, K_size, 3)  # (K_number x K_size x K_size x Channels)
    
    feature_vectors = []
    
    # Use tqdm to track the progress of the loop
    for img in tqdm(images, desc="Processing Images"):
        feature_maps = cnn_layer(img, filters, pad=pad, step=step)
        
        relu_maps = relu_layer(feature_maps)
        
        pooled_maps = pooling_layer(relu_maps, size=2, step=2)
        
        fc_output = fully_connected_layer(pooled_maps)
        
        feature_vectors.append(fc_output)
    
    feature_vectors = np.array(feature_vectors)
    
    # Print the resulting feature vectors
    print("Feature vectors from the CNN:")
    print(feature_vectors)

    
if __name__ == "__main__":
    main()

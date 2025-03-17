[1mdiff --git a/src/main/python/model/CNN.py b/.gitattributes[m
[1msimilarity index 100%[m
[1mrename from src/main/python/model/CNN.py[m
[1mrename to .gitattributes[m
[1mdiff --git a/README.md b/README.md[m
[1mindex 6d92485..88a92b0 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -2,5 +2,14 @@[m
 Comparing faces using vectorization[m
 [m
 # image-source[m
[31m-I don't own these images, they were made public here:[m
[31m-https://archive.org/details/attractive-faces[m
\ No newline at end of file[m
[32m+[m
[32m+[m[32mIf you use this dataset in your research work, please cite[m
[32m+[m
[32m+[m[32m"Challenges in Representation Learning: A report on three machine learning[m
[32m+[m[32mcontests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B[m
[32m+[m[32mHamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,[m
[32m+[m[32mX Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,[m
[32m+[m[32mM Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and[m
[32m+[m[32mY. Bengio. arXiv 2013.[m
[32m+[m
[32m+[m[32mSee fer2013.bib for a bibtex entry.[m
[1mdiff --git a/src/main/python/model/Data_loader.py b/src/main/python/model/Data_loader.py[m
[1mnew file mode 100644[m
[1mindex 0000000..9bee123[m
[1m--- /dev/null[m
[1m+++ b/src/main/python/model/Data_loader.py[m
[36m@@ -0,0 +1,22 @@[m
[32m+[m[32mimport pandas as pd[m
[32m+[m[32mimport numpy as np[m
[32m+[m[32mfrom sklearn.model_selection import train_test_split[m
[32m+[m
[32m+[m[32mdef load_and_preprocess_test_data(csv_path, target_size=(48, 48)):[m
[32m+[m[32m    df = pd.read_csv(csv_path)[m
[32m+[m[32m    images = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ') for pixels in df['pixels']])[m
[32m+[m[32m    images = images.reshape(-1, 48, 48, 1)[m
[32m+[m
[32m+[m
[32m+[m[32m    return images[m
[32m+[m
[32m+[m[32mdef load_and_preprocess_train_data(csv_path, target_size=(48, 48)):[m
[32m+[m[32m    df = pd.read_csv(csv_path)[m
[32m+[m[32m    images = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ') for pixels in df['pixels']])[m
[32m+[m[32m    images = images.reshape(-1, 48, 48, 1)[m
[32m+[m
[32m+[m[32m    # Load the labels (8 emotions)[m
[32m+[m[32m    labels = pd.get_dummies(df['emotion']).values  # Assuming 'emotion' is the column name[m
[32m+[m
[32m+[m[32m    return images, labels[m
[32m+[m
[1mdiff --git a/src/main/python/model/Main.py b/src/main/python/model/Main.py[m
[1mindex ab13e16..b7232ee 100644[m
[1m--- a/src/main/python/model/Main.py[m
[1m+++ b/src/main/python/model/Main.py[m
[36m@@ -1,252 +1,211 @@[m
 import os[m
 import matplotlib.pyplot as plt[m
 from PIL import Image[m
[32m+[m[32mfrom Data_loader import load_and_preprocess_test_data[m
[32m+[m
 import numpy as np[m
 from tqdm import tqdm[m
[32m+[m[32mimport pandas as pd[m
[32m+[m[32mimport cv2[m
 [m
[31m-def load_and_preprocess_data(image_folder, target_size, pad):[m
[32m+[m[32mdef visualize_images(images, num_images=5):[m
     """[m
[31m-    Load and preprocess images from the given folder.[m
[32m+[m[32m    Visualize the first few images from the dataset.[m
[32m+[m[41m    [m
     Args:[m
[31m-        image_folder (str): Path to the folder containing the images.[m
[31m-        target_size (tuple): The target size to which all images will be resized (default: (224, 224)).[m
[31m-    Returns:[m
[31m-        np.ndarray: Array of preprocessed images.[m
[32m+[m[32m        images (np.ndarray): Array of preprocessed images.[m
[32m+[m[32m        num_images (int): Number of images to visualize.[m
     """[m
[31m-    processed_images = [][m
[31m-[m
[31m-    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][m
[32m+[m[32m    for i in range(num_images):[m
[32m+[m[32m        plt.figure(figsize=(2, 2))[m
[32m+[m[32m        plt.imshow(images[i].reshape(48, 48), cmap='gray')  # Reshape to (48, 48) for grayscale display[m
[32m+[m[32m        plt.title(f'Image {i + 1}')[m
[32m+[m[32m        plt.axis('off')[m
[32m+[m[32m        plt.show()[m
 [m
[31m-    for image_name in image_files:[m
[31m-        image_path = os.path.join(image_folder, image_name)[m
[31m-        [m
[31m-        img = load_image(image_path)[m
[31m-        [m
[31m-        img = preprocess_image(img, target_size, pad)[m
[31m-        [m
[31m-        if img is not None:[m
[31m-            processed_images.append(img)[m
[31m-    [m
[31m-    return np.array(processed_images)[m
[31m-[m
[31m-def load_image(image_path):[m
[32m+[m[32mdef visualize_feature_maps(feature_maps, num_maps=5):[m
     """[m
[31m-    Load an image from the specified path.[m
[32m+[m[32m    Visualize the feature maps after applying the CNN layers.[m
[32m+[m[41m    [m
[32m+[m[32m    Args:[m
[32m+[m[32m        feature_maps (np.ndarray): The feature maps after CNN layers.[m
[32m+[m[32m        num_maps (int): Number of feature maps to visualize.[m
     """[m
[31m-    try:[m
[31m-        img = Image.open(image_path)[m
[31m-        return img[m
[31m-    except Exception as e:[m
[31m-        print(f"Error loading image {image_path}: {e}")[m
[31m-        return None[m
[32m+[m[32m    # If the feature maps have only one channel, reshape to 3D for uniformity[m
[32m+[m[32m    if len(feature_maps.shape) == 2:  # If the output is 2D[m
[32m+[m[32m        feature_maps = feature_maps[:, :, np.newaxis]  # Add a singleton dimension for channels[m
 [m
[31m-def preprocess_image(img, target_size , pad_width):[m
[31m-    """[m
[31m-    Preprocess the image by resizing, normalizing, and adding padding.[m
[31m-    """[m
[31m-    if img is not None:[m
[31m-        img = img.resize(target_size)[m
[31m-        img = img.convert('RGB')  # Ensures the image is in RGB format[m
[31m-        [m
[31m-        # Add padding of pad_width (default 1) around the image[m
[31m-        img_padded = np.pad(np.array(img), ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)[m
[31m-        [m
[31m-        return img_padded[m
[31m-    else:[m
[31m-        return None[m
[31m-import numpy as np[m
[32m+[m[32m    num_feature_maps = feature_maps.shape[-1]  # Number of channels (filters)[m
[32m+[m[41m    [m
[32m+[m[32m    for i in range(min(num_maps, num_feature_maps)):[m
[32m+[m[32m        plt.figure(figsize=(2, 2))[m
[32m+[m[32m        plt.imshow(feature_maps[:, :, i], cmap='gray')  # Display each feature map as grayscale[m
[32m+[m[32m        plt.title(f'Feature Map {i + 1}')[m
[32m+[m[32m        plt.axis('off')[m
[32m+[m[32m        plt.show()[m
 [m
 [m
[31m-def convolution_2d(image, filter, pad, step):[m
[32m+[m[32mdef convolution_2d(image, filter, step):[m
     """[m
     Perform a 2D convolution operation on an image with a given filter.[m
[31m-    [m
[31m-    Args:[m
[31m-        image (np.ndarray): The input image to be convolved.[m
[31m-        filter (np.ndarray): The filter (kernel) to apply.[m
[31m-        pad (int): The padding applied to the image.[m
[31m-        step (int): The step size (stride) for the convolution.[m
[31m-    [m
[31m-    Returns:[m
[31m-        np.ndarray: The resulting image after convolution.[m
     """[m
     k_size = filter.shape[0][m
[31m-[m
[31m-    width_out = int((image.shape[0] - k_size + 2 * pad) / step + 1)[m
[31m-    height_out = int((image.shape[1] - k_size + 2 * pad) / step + 1)[m
[31m-[m
[31m-    output_image = np.zeros((width_out - 2 * pad, height_out - 2 * pad))[m
[31m-[m
[31m-    for i in range(image.shape[0] - k_size + 1):[m
[31m-        for j in range(image.shape[1] - k_size + 1):[m
[32m+[m[32m    width_out = int((image.shape[0] - k_size) / step + 1)[m
[32m+[m[32m    height_out = int((image.shape[1] - k_size) / step + 1)[m
[32m+[m[32m    output_image = np.zeros((width_out, height_out))[m
[32m+[m[41m    [m
[32m+[m[32m    for i in range(0, width_out, step):[m
[32m+[m[32m        for j in range(0, height_out, step):[m
             patch_from_image = image[i:i+k_size, j:j+k_size][m
             output_image[i, j] = np.sum(patch_from_image * filter)[m
[31m-[m
[32m+[m[41m    [m
     return output_image[m
 [m
[31m-[m
[31m-def cnn_layer(image_volume, filter, pad=1, step=1):[m
[32m+[m[32mdef cnn_layer(image_volume, filter, step=1):[m
[32m+[m[32m    """[m
[32m+[m[32m    Apply a convolutional layer on the input image volume with a given filter.[m
     """[m
[31m-    Apply a convolutional layer on the input image volume with a given filter set.[m
[32m+[m[32m    k_size = filter.shape[0][m
[32m+[m[32m    width_out = int((image_volume.shape[0] - k_size) / step + 1)[m
[32m+[m[32m    height_out = int((image_volume.shape[1] - k_size) / step + 1)[m
     [m
[31m-    Args:[m
[31m-        image_volume (np.ndarray): The input image volume (height x width x depth).[m
[31m-        filter (np.ndarray): The filter set (number_of_filters x filter_height x filter_width x input_depth).[m
[31m-        pad (int): The padding applied to the image (default is 1).[m
[31m-        step (int): The step size (stride) for the convolution (default is 1).[m
[32m+[m[32m    # Only one depth dimension since it's grayscale[m
[32m+[m[32m    feature_map = np.zeros((width_out, height_out))[m
     [m
[31m-    Returns:[m
[31m-        np.ndarray: The feature maps after applying the convolutional layer.[m
[31m-    """[m
[31m-    image = np.zeros((image_volume.shape[0] + 2 * pad, image_volume.shape[1] + 2 * pad, image_volume.shape[2]))[m
[31m-[m
[31m-    for p in range(image_volume.shape[2]):[m
[31m-        image[:, :, p] = np.pad(image_volume[:, :, p], (pad, pad), mode='constant', constant_values=0)[m
[31m-[m
[31m-    k_size = filter.shape[1][m
[31m-    depth_out = filter.shape[0][m
[31m-    width_out = int((image_volume.shape[0] - k_size + 2 * pad) / step + 1)[m
[31m-    height_out = int((image_volume.shape[1] - k_size + 2 * pad) / step + 1)[m
[31m-[m
[31m-    feature_maps = np.zeros((width_out, height_out, depth_out))[m
[31m-[m
[31m-    n_filters = filter.shape[0][m
[31m-[m
[31m-    for i in range(n_filters):[m
[31m-        convolved_image = np.zeros((width_out, height_out))[m
[31m-[m
[31m-        for j in range(image.shape[-1]):[m
[31m-            convolved_image += convolution_2d(image[:, :, j], filter[i, :, :, j], pad, step)[m
[31m-        [m
[31m-        feature_maps[:, :, i] = convolved_image[m
[31m-[m
[31m-    return feature_maps[m
[31m-[m
[31m-[m
[31m-def image_pixels_255(maps):[m
[31m-[m
[31m-    """[m
[31m-    Replaces values in an image over 255 with 255[m
[31m-    """[m
[31m-    r = np.zeros(maps.shape)[m
[32m+[m[32m    feature_map = convolution_2d(image_volume, filter, step)[m
[32m+[m[41m    [m
[32m+[m[32m    return feature_map[m
 [m
[31m-    for c in range(r.shape[2]):[m
[31m-        for i in range(r.shape[0]):[m
[31m-            for j in range(r.shape[1]):[m
[31m-                if maps[i, j, c] <= 255:[m
[31m-                    r[i, j, c] = maps[i, j, c][m
[31m-                else:[m
[31m-                    r[i, j, c] = 255[m
[31m-    return r[m
 [m
 def relu_layer(maps):[m
     """[m
[31m-  [m
[31m-    Args:[m
[31m-        maps (np.ndarray): The input feature maps, which can be the output of a convolutional layer.[m
[31m-    [m
[31m-    Returns:[m
[31m-        np.ndarray: The feature maps after applying the ReLU activation function, [m
[31m-                    where all negative values are replaced with zeros.[m
[32m+[m[32m    Apply ReLU activation function.[m
     """[m
[31m-    r = np.zeros_like(maps)[m
[31m-    [m
[31m-    result = np.where(maps > r, maps, r)[m
[31m-    [m
[31m-    return result[m
[32m+[m[32m    return np.maximum(0, maps)[m
 [m
 def pooling_layer(maps, size=2, step=2):[m
     """[m
[31m-    Apply the max pooling operation to the input feature maps.[m
[31m-[m
[31m-    Args:[m
[31m-        maps (np.ndarray): The input feature maps (height x width x depth).[m
[31m-        size (int): The size of the pooling filter (default is 2x2).[m
[31m-        step (int): The stride (step size) for the pooling operation (default is 2).[m
[31m-[m
[31m-    Returns:[m
[31m-        np.ndarray: The pooled feature maps with reduced spatial dimensions.[m
[32m+[m[32m    Apply max pooling.[m
     """[m
     width_out = int((maps.shape[0] - size) / step + 1)[m
     height_out = int((maps.shape[1] - size) / step + 1)[m
[32m+[m[32m    pooling_image = np.zeros((width_out, height_out))[m
[32m+[m[41m    [m
[32m+[m[32m    for i in range(0, width_out, step):[m
[32m+[m[32m        for j in range(0, height_out, step):[m
[32m+[m[32m            patch_from_image = maps[i:i+size, j:j+size][m
[32m+[m[32m            pooling_image[i, j] = np.max(patch_from_image)[m
[32m+[m[41m    [m
[32m+[m[32m    return pooling_image[m
 [m
[31m-    pooling_image = np.zeros((width_out, height_out, maps.shape[2]))[m
[32m+[m[32mdef fully_connected_layer(feature_map, output_size=7):[m
[32m+[m[32m    """[m
[32m+[m[32m    Fully connected layer for multi-class classification (7 emotions).[m
[32m+[m[32m    """[m
[32m+[m[32m    flattened = feature_map.flatten()[m
[32m+[m[32m    weights = np.random.randn(flattened.shape[0], output_size)[m
[32m+[m[32m    output = np.dot(flattened, weights)[m
[32m+[m[32m    return output[m
 [m
[31m-    for c in range(maps.shape[2]):[m
[31m-        ii = 0[m
[31m-        for i in range(0, maps.shape[0] - size + 1, step):[m
[31m-            jj = 0[m
[31m-            for j in range(0, maps.shape[1] - size + 1, step):[m
[31m-                patch_from_image = maps[i:i+size, j:j+size, c][m
[31m-                pooling_image[ii, jj, c] = np.max(patch_from_image)[m
[31m-                jj += 1[m
[31m-            ii += 1[m
[32m+[m[32mdef softmax(logits):[m
[32m+[m[32m    """[m
[32m+[m[32m    Apply softmax to logits to get probabilities.[m
[32m+[m[32m    """[m
[32m+[m[32m    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability[m
[32m+[m[32m    return exp_logits / np.sum(exp_logits)[m
 [m
[31m-    return pooling_image[m
 [m
[31m-def fully_connected_layer(feature_map, output_size=10):[m
[32m+[m[32mdef apply_cnn_blocks(image, filters_list, step=1, pool_size=2, pool_step=2):[m
     """[m
[31m-    A simple fully connected layer that takes in the feature map and outputs a vector of size `output_size`.[m
[31m-    [m
[32m+[m[32m    Applies multiple convolutional layers sequentially with ReLU activation and pooling.[m
[32m+[m
     Args:[m
[31m-        feature_map (np.ndarray): Flattened feature map (a 1D array of features).[m
[31m-        output_size (int): The number of output classes (default is 10).[m
[31m-    [m
[32m+[m[32m        image (np.ndarray): The input image.[m
[32m+[m[32m        filters_list (list of np.ndarray): List of filters for each convolutional layer.[m
[32m+[m[32m        step (int): Stride for convolution.[m
[32m+[m[32m        pool_size (int): Pooling window size.[m
[32m+[m[32m        pool_step (int): Stride for pooling.[m
[32m+[m
     Returns:[m
[31m-        np.ndarray: The final output vector of size `output_size`.[m
[32m+[m[32m        np.ndarray: The output feature map after the CNN layers.[m
     """[m
[31m-    flattened = feature_map.flatten()[m
[31m-    [m
[31m-    #TODO train weights[m
[31m-    weights = np.random.randn(flattened.shape[0], output_size)[m
[31m-    [m
[31m-    # Output vector[m
[31m-    output = np.dot(flattened, weights)[m
[31m-    [m
[32m+[m[32m    output = image[m[41m  [m
[32m+[m
[32m+[m[32m    for filter in filters_list:[m
[32m+[m[32m        output = cnn_layer(output, filter, step=step)  # Convolution[m
[32m+[m[32m        output = relu_layer(output)                     # ReLU activation[m
[32m+[m[32m        output = pooling_layer(output, size=pool_size, step=pool_step)  # Pooling[m
[32m+[m
     return output[m
 [m
 def main():[m
[31m-    # Hyperparameters:[m
[31m-    K_number = 2    # number of filters (kernels) denoted as K_number[m
[31m-    K_size = 3      # size of filters (spatial dimension) denoted as K_size[m
[31m-    step = 1        # step for sliding (also known as stride) denoted as Step[m
[31m-    pad = 1         # processing edges by zero-padding parameter denoted as Pad[m
[31m-    target_size = (224, 224)  # image size[m
[32m+[m
[32m+[m[32m    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][m
[32m+[m[32m    num_images = 10  # how many images i take[m
[32m+[m
[32m+[m[32m    # Hyperparameters[m
[32m+[m[32m    K_number = 2    # Number of filters per layer[m
[32m+[m[32m    K_size = 3      # Filter size[m
[32m+[m[32m    step = 1        # Convolution stride[m
[32m+[m[32m    target_size = (48, 48)  # Image size[m
[32m+[m[32m    num_layers = 3  # Number of convolutional blocks[m
 [m
     script_dir = os.path.dirname(os.path.realpath(__file__))[m
[31m-    project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))[m
[31m-    image_folder = os.path.join(project_root, 'src', 'main', 'resources', 'faces')[m
[32m+[m[32m    data_folder = os.path.join(script_dir, '..', '..', 'resources', 'faces')[m
     [m
[31m-    if not os.path.exists(image_folder):[m
[31m-        print(f"The directory {image_folder} does not exist.")[m
[32m+[m[32m    test_csv_path = os.path.join(data_folder, 'test.csv')[m
[32m+[m
[32m+[m[32m    if not os.path.exists(test_csv_path):[m
[32m+[m[32m        print(f"The file {test_csv_path} does not exist.")[m
         return[m
[31m-    [m
[32m+[m
     # Load and preprocess data[m
[31m-    images = load_and_preprocess_data(image_folder, target_size, pad)[m
[31m-    [m
[31m-    # TODO train filters[m
[31m-    filters = np.random.randn(K_number, K_size, K_size, 3)  # (K_number x K_size x K_size x Channels)[m
[31m-    [m
[32m+[m[32m    images = load_and_preprocess_test_data(test_csv_path, target_size)[m
[32m+[m
[32m+[m[32m    images = images[:num_images][m
[32m+[m
[32m+[m[32m    # Initialize random filters for each layer[m
[32m+[m[32m    filters_list = [np.random.randn(K_size, K_size) for _ in range(num_layers)]  # Only 2D filters for grayscale[m
[32m+[m
     feature_vectors = [][m
[31m-    [m
[31m-    # Use tqdm to track the progress of the loop[m
[31m-    for img in tqdm(images, desc="Processing Images"):[m
[31m-        feature_maps = cnn_layer(img, filters, pad=pad, step=step)[m
[31m-        [m
[31m-        relu_maps = relu_layer(feature_maps)[m
[31m-        [m
[31m-        pooled_maps = pooling_layer(relu_maps, size=2, step=2)[m
[31m-        [m
[31m-        fc_output = fully_connected_layer(pooled_maps)[m
[31m-        [m
[31m-        feature_vectors.append(fc_output)[m
[31m-    [m
[32m+[m
[32m+[m[32m    for idx, img in enumerate(tqdm(images, desc="Processing Images")):[m
[32m+[m[32m        # Create a figure with 1 row and 2 columns[m
[32m+[m[32m        fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # Two plots: image and probability bar chart[m
[32m+[m
[32m+[m[32m        # Plot the image[m
[32m+[m[32m        ax[0].imshow(img.reshape(48, 48), cmap='gray')  # Reshape to (48, 48) for grayscale display[m
[32m+[m[32m        ax[0].set_title(f'Image {idx + 1}')[m
[32m+[m[32m        ax[0].axis('off')  # Hide axes[m
[32m+[m
[32m+[m[32m        # Apply CNN blocks[m
[32m+[m[32m        feature_maps = apply_cnn_blocks(img, filters_list, step=step)[m
[32m+[m
[32m+[m[32m        # Fully connected layer output and softmax probabilities[m
[32m+[m[32m        fc_output = fully_connected_layer(feature_maps)  # Fully connected layer[m
[32m+[m[32m        probabilities = softmax(fc_output)  # Apply softmax to get probabilities[m
[32m+[m
[32m+[m[32m        feature_vectors.append(probabilities)  # Append probabil
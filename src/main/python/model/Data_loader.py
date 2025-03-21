import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path, usage_type, target_size=(48, 48)):
    """
    Loads and preprocesses the data from the CSV file. Filters based on the 'Usage' column.
    
    Parameters:
    - csv_path: Path to the CSV file.
    - usage_type: The type of usage ('Training', 'PrivateTest', etc.) to filter by.
    - target_size: Target size of images for resizing (default is 48x48).
    
    Returns:
    - images: Preprocessed and normalized image data with shape (num_samples, height, width).
    - labels: Preprocessed label data (one-hot encoded for training data).
    """
    df = pd.read_csv(csv_path)
    
    # Filter rows based on the usage type
    df = df[df[' Usage'] == usage_type]
    
    # Convert pixel strings to arrays of uint8, then reshape to image dimensions
    images = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ') for pixels in df[' pixels']])
    images = images.reshape(-1, target_size[0], target_size[1])
    
    # Normalize images to range [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # One-hot encode the emotion labels
    labels = pd.get_dummies(df['emotion']).values
    
    return images, labels

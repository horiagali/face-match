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
    - images: Preprocessed image data with shape (num_samples, height, width).
    - labels: Preprocessed label data (dummy labels for non-training data).
    """
    df = pd.read_csv(csv_path)
    
    df = df[df[' Usage'] == usage_type]
    
    images = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ') for pixels in df[' pixels']])
    
    images = images.reshape(-1, target_size[0], target_size[1])

    labels = pd.get_dummies(df['emotion']).values
    
    return images, labels

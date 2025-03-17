import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_test_data(csv_path, target_size=(48, 48)):
    df = pd.read_csv(csv_path)
    images = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ') for pixels in df['pixels']])
    images = images.reshape(-1, 48, 48, 1)


    return images

def load_and_preprocess_train_data(csv_path, target_size=(48, 48)):
    df = pd.read_csv(csv_path)
    images = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ') for pixels in df['pixels']])
    images = images.reshape(-1, 48, 48, 1)

    # Load the labels (8 emotions)
    labels = pd.get_dummies(df['emotion']).values  # Assuming 'emotion' is the column name

    return images, labels


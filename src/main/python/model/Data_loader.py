import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

def load_and_preprocess_data(csv_path, usage_type, target_size=(48, 48)):
    """
    Loads and preprocesses the data from the CSV file. Filters based on the 'Usage' column.
    
    Parameters:
    - csv_path: Path to the CSV file.
    - usage_type: The type of usage ('Training', 'PrivateTest', etc.) to filter by.
    - target_size: Target size of images for resizing (default is 48x48).
    
    Returns:
    - images: Preprocessed and normalized image data with shape (num_samples, height, width, channels).
    - labels: Preprocessed label data (one-hot encoded for training data).
    """
    df = pd.read_csv(csv_path)
    
    # Filter rows based on the usage type
    df = df[df[' Usage'] == usage_type]
    
    # Convert pixel strings to arrays of uint8
    images = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ') for pixels in df[' pixels']])
    
    # Reshape images to the target size and add a channel dimension (grayscale)
    images = images.reshape(-1, target_size[0], target_size[1], 1)  # (num_samples, height, width, channels)
    
    # Normalize images to range [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # One-hot encode the emotion labels
    labels = pd.get_dummies(df['emotion']).values
    
    return images, labels


def load_model(filename):
    """
    Loads a saved EmotionCNN model from a .npz file.
    """
    data = np.load(filename, allow_pickle=True)
    hyperparams = json.loads(data['hyperparams'].item())

    model = EmotionCNN(
        filters_list=data['filters'].tolist(),
        biases_list=data['biases'].tolist(),
        fc_weights=data['fc_weights'].tolist(),
        fc_bias=data['fc_bias'].tolist(),
        output_weights=data['output_weights'],
        output_bias=data['output_bias'],
        gamma=data['gamma'].tolist(),
        beta=data['beta'].tolist(),
        step=int(data['step']),
        pool_size=int(data['pool_size']),
        pool_step=int(data['pool_step']),
        dropout_rate_conv=float(data['dropout_rate_conv']),
        dropout_rate_fc=float(data['dropout_rate_fc'])
    )

    return model, hyperparams

def save_model(model, hyperparams, filename):
    """
    Saves the model parameters and hyperparameters to a .npz file.
    """
    hyperparams_json = json.dumps(hyperparams)
    np.savez(filename,
             filters=np.array(model.filters_list, dtype=object),
             biases=np.array(model.biases_list, dtype=object),
             fc_weights=np.array(model.fc_weights, dtype=object),
             fc_bias=np.array(model.fc_bias, dtype=object),
             output_weights=model.output_weights,
             output_bias=model.output_bias,
             gamma=np.array(model.gamma, dtype=object),
             beta=np.array(model.beta, dtype=object),
             step=model.step,
             pool_size=model.pool_size,
             pool_step=model.pool_step,
             dropout_rate_conv=model.dropout_rate_conv,
             dropout_rate_fc=model.dropout_rate_fc,
             hyperparams=hyperparams_json)
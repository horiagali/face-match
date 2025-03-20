import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Data_loader import load_and_preprocess_data
from emotion_cnn import EmotionCNN

def load_model(filename):
    """Loads trained model parameters and hyperparameters from a .npz file."""
    data = np.load(filename, allow_pickle=True)
    hyperparams = json.loads(data['hyperparams'].item())

    # Convert filters and biases to numpy arrays
    filters = [np.array(f) for f in data['filters']]
    biases = [np.array(b) for b in data['biases']]

    model = EmotionCNN(
        filters_list=filters,  # Ensure filters are numpy arrays
        biases_list=biases,    # Ensure biases are numpy arrays
        fc_weights=data['fc_weights'],
        fc_bias=data['fc_bias'],
        step=hyperparams.get("step", 1),
        pool_size=hyperparams.get("pool_size", 2),
        pool_step=hyperparams.get("pool_step", 2)
    )
    
    return model, hyperparams



def main():
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Load model parameters from file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, '..', '..','..', '..', 'model_fold_5.npz')
    
    # Load model parameters from file
    cnn_model, hyperparams = load_model(model_path)

    # Set parameters from the model
    target_size = (48, 48)

    # Load test dataset
    data_folder = os.path.join(script_dir, '..', '..', 'resources', 'faces')
    test_csv_path = os.path.join(data_folder, 'data.csv')
    images_test, labels_test = load_and_preprocess_data(test_csv_path, usage_type='PrivateTest', target_size=target_size)

    # Select the first 10 images dynamically
    num_images_to_visualize = min(100, len(images_test))

    images_to_visualize = images_test[:num_images_to_visualize]
    labels_to_visualize = labels_test[:num_images_to_visualize]

    correct_predictions = 0

    for idx, (img, true_label) in enumerate(zip(images_to_visualize, labels_to_visualize)):
        # Pass image through the CNN model
        probabilities = cnn_model.forward(img)  
        predicted_label = np.argmax(probabilities)
        correct_predictions += (predicted_label == np.argmax(true_label))

        # Plot image with predicted probabilities and true label
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(img.reshape(48, 48), cmap='gray')
        ax[0].set_title(f'True: {emotions[np.argmax(true_label)]}\nPredicted: {emotions[predicted_label]}')
        ax[0].axis('off')

        ax[1].bar(emotions, probabilities * 100, color='blue')
        ax[1].set_title('Predicted Probabilities')
        ax[1].set_ylabel('Probability (%)')
        ax[1].set_ylim([0, 100])

        plt.tight_layout()
        plt.show()

    # Print probabilities
    print(f"Image {idx + 1} probabilities:")
    for emotion, prob in zip(emotions, probabilities):
        print(f"  {emotion}: {prob * 100:.2f}%")
    print()


if __name__ == "__main__":
    main()

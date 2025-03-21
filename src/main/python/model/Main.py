import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Data_loader import load_and_preprocess_data
from emotion_cnn import EmotionCNN

def load_model(filename, K_size=3, num_layers=4, output_size=7):
    """Loads model parameters and hyperparameters from a .npz file, but always uses random filters."""
    data = np.load(filename, allow_pickle=True)
    hyperparams = json.loads(data['hyperparams'].item())

    # Load filters and biases for each convolutional layer
    filters = [np.array(f) for f in data['filters']]
    biases = [np.array(b) for b in data['biases']]

    # Initialize the model with loaded filters and biases
    model = EmotionCNN(
        K_size=K_size,
        num_layers=num_layers,
        output_size=output_size
    )

    # Assign the loaded filters and biases to the model
    for i in range(num_layers):
        model.conv_params[i]["W"] = filters[i]
        model.conv_params[i]["b"] = biases[i]

    # Load the fully connected layer weights and biases
    model.fc1_W = data['fc_weights']
    model.fc1_b = data['fc_bias']
    
    # Return the model with loaded parameters
    return model



def main():
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Load model parameters from file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, '..', '..','..', '..', 'model_fold_5.npz')
    
    # Load model parameters from file
    cnn_model = load_model(model_path)

    # Set parameters from the model
    target_size = (48, 48)

    # Load test dataset
    data_folder = os.path.join(script_dir, '..', '..', 'resources', 'faces')
    test_csv_path = os.path.join(data_folder, 'data.csv')
    images_test, labels_test = load_and_preprocess_data(test_csv_path, usage_type='PrivateTest', target_size=target_size)

    # Select the first 10 images dynamically
    num_images = 5
    num_images_to_visualize = min(num_images, len(images_test))

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

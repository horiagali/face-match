import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from Data_loader import load_and_preprocess_data  # Corrected import

from emotion_cnn import EmotionCNN



def main():
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    num_images_to_visualize = 2  # Number of images to visualize and make predictions

    # Hyperparameters
    K_number = 2
    K_size = 3
    step = 1
    target_size = (48, 48)
    num_layers = 3
    output_size = 7  # Number of emotions

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, '..', '..', 'resources', 'faces')
    test_csv_path = os.path.join(data_folder, 'data.csv')

    if not os.path.exists(test_csv_path):
        print(f"The file {test_csv_path} does not exist.")
        return

    # Load and preprocess the test data
    images_test, _ = load_and_preprocess_data(test_csv_path, usage_type='PrivateTest', target_size=target_size)
    
    # Only visualize the first 10 images
    images_to_visualize = images_test[:num_images_to_visualize]

    # Initialize random filters and biases for each layer
    filters_list = [np.random.randn(K_size, K_size) for _ in range(num_layers)]
    biases_list = [np.random.randn() for _ in range(num_layers)]  

    # Initialize the EmotionCNN model
    cnn_model = EmotionCNN(filters_list, biases_list, np.random.randn(filters_list[0].size * len(filters_list), output_size), np.random.randn(output_size))

    # Visualize images and make predictions 
    feature_vectors = []

    for idx, img in enumerate(images_to_visualize):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(img.reshape(48, 48), cmap='gray')
        ax[0].set_title(f'Image {idx + 1}')
        ax[0].axis('off')

        # Pass image through the CNN model
        feature_maps, _ = cnn_model.apply_cnn_blocks(img)
        fc_output = cnn_model.fully_connected_layer(feature_maps)
        probabilities = cnn_model.softmax(fc_output)

        feature_vectors.append(probabilities)

        # Plot the predicted probabilities for each emotion
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

    feature_vectors = np.array(feature_vectors)

    # Load the full PrivateTest dataset for final predictions
    icml_csv_path = os.path.join(script_dir, '..', '..', 'resources', 'faces', 'icml_face_data.csv')
    images_all, labels_all = load_and_preprocess_data(icml_csv_path, usage_type='Training', target_size=target_size)

    # Initialize counters for accuracy calculation
    correct_predictions = 0

    # Make predictions on all the images in icml_face_data.csv
    for img, true_label in tqdm(zip(images_all, labels_all), total=len(images_all), desc="Processing Images"):
        # Pass image through the CNN model for final predictions
        feature_maps, _ = cnn_model.apply_cnn_blocks(img)
        fc_output = cnn_model.fully_connected_layer(feature_maps)
        probabilities = cnn_model.softmax(fc_output)

        predicted_label = emotions[np.argmax(probabilities)]
        if predicted_label == true_label:
            correct_predictions += 1

    # Calculate precision (accuracy)
    precision = (correct_predictions / len(labels_all)) * 100
    print(f"Model precision on PrivateTest data: {precision:.2f}%")

if __name__ == "__main__":
    main()

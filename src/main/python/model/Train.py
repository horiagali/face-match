import os
import json
import numpy as np
from Data_loader import load_and_preprocess_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from emotion_cnn import EmotionCNN
from tqdm import tqdm  

def save_model(model, hyperparams, filename):
    """
    Saves the trained model parameters and hyperparameters to a .npz file.
    The hyperparameters are stored as a JSON string.
    """
    hyperparams_json = json.dumps(hyperparams)
    # Save convolution filters, biases, and fully connected parameters.
    np.savez(filename,
             filters=np.array(model.filters_list, dtype=object),
             biases=np.array(model.biases_list, dtype=object),
             fc_weights=model.fc_weights,
             fc_bias=model.fc_bias,
             hyperparams=hyperparams_json)

def load_model(filename):
    """
    Loads model parameters and hyperparameters from the given .npz file.
    Returns the reconstructed EmotionCNN model and hyperparameters dictionary.
    """
    data = np.load(filename, allow_pickle=True)
    hyperparams = json.loads(data['hyperparams'].item())
    model = EmotionCNN(filters_list=data['filters'].tolist(),
                       biases_list=data['biases'].tolist(),
                       fc_weights=data['fc_weights'],
                       fc_bias=data['fc_bias'])
    return model, hyperparams

def k_fold_cross_validation(csv_path, k=5, epochs=10, learning_rate=0.01, 
                            K_size=3, num_layers=3, output_size=7):
    images, labels = load_and_preprocess_data(csv_path, "Training")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []



    
    # Wrap the kf.split(images) iterator with tqdm to add a progress bar for fold iterations
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(images), total=k, desc="K-Fold Cross Validation", unit="fold"), 1):
        print(f"\n--- Fold {fold}/{k} ---")
        X_train, X_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Initialize random convolution filters and biases for each conv layer
        filters_list = [np.random.randn(K_size, K_size) for _ in range(num_layers)]
        biases_list = [np.random.randn() for _ in range(num_layers)]

        # Now initialize the actual fully connected layer with the correct input size
        fc_weights = np.random.randn(16, output_size)
        fc_bias = np.random.randn(output_size)

        # Initialize the EmotionCNN model with all parameters
        machine = EmotionCNN(filters_list, biases_list, fc_weights, fc_bias)
        
        # Training phase with a progress bar using tqdm
        train_losses, test_losses = [], []
        for epoch in tqdm(range(epochs), desc=f"Training Fold {fold}", unit="epoch"):
            epoch_train_loss = 0
            # Training phase: update model parameters via backpropagation
            for img, label in zip(X_train, y_train):
                # Step 1: Forward pass to get predictions
                preds = machine.forward(img)
                
                # Step 2: Calculate the loss
                one_hot = np.zeros(output_size)
                one_hot[label] = 1
                loss = machine.cross_entropy_loss(preds, one_hot)
                
                # Step 3: Backpropagation to update weights
                machine.backpropagate(img, label, learning_rate)
                
                epoch_train_loss += loss
            
            avg_train_loss = epoch_train_loss / len(X_train)
            train_losses.append(avg_train_loss)
            
            # Testing phase: compute loss on validation set
            epoch_test_loss = 0
            for img, label in zip(X_val, y_val):
                preds = machine.forward(img)
                one_hot = np.zeros(output_size)
                one_hot[label] = 1
                loss_test = machine.cross_entropy_loss(preds, one_hot)
                epoch_test_loss += loss_test
            avg_test_loss = epoch_test_loss / len(X_val)
            test_losses.append(avg_test_loss)

        
        # Plot the training and test loss curves for this fold
        plt.figure()
        plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
        plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs. Epochs for Fold {fold}")
        plt.legend()
        plt.show()
        
        # Evaluate the model on validation data to compute accuracy
        predictions = []
        for img in X_val:
            output = machine.forward(img)
            predictions.append(np.argmax(output))
        accuracy = accuracy_score(np.argmax(y_val, axis=1), predictions)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy * 100:.2f}%")
        
        # Save the trained model along with its hyperparameters for this fold
        hyperparams = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "K_size": K_size,
            "num_layers": num_layers,
            "output_size": output_size
        }
        save_model(machine, hyperparams, f"model_fold_{fold}.npz")
    
    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage Accuracy over {k} folds: {avg_accuracy * 100:.2f}%")

def main():
    
    # Get the absolute path to the CSV file based on the script's location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, '..', '..', 'resources', 'faces')
    csv_path = os.path.join(data_folder, 'data.csv')
    # Perform K-fold cross-validation with training, loss-curve plotting, and model saving
    k_fold_cross_validation(csv_path, k=5, epochs=10, learning_rate=0.01,
                            K_size=3, num_layers=3, output_size=7)


if __name__ == "__main__":
    main()

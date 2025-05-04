import os
import numpy as np
from Data_loader import load_and_preprocess_data
from Data_loader import save_model

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from emotion_cnn import EmotionCNN
from tqdm import tqdm

def cross_entropy_loss(predictions, labels):
        """
        Compute the cross-entropy loss.

        predictions: np.array of shape (batch_size, num_classes)
            The predicted probabilities for each class.
        labels: np.array of shape (batch_size, num_classes)
            The true one-hot encoded labels.

        Returns:
        loss: scalar
            The computed cross-entropy loss for the batch.
        """
        # Clip predictions to prevent log(0)
        epsilon = 1e-15  # small value to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1. - epsilon)

        # Compute the cross-entropy loss
        loss = -np.sum(labels * np.log(predictions)) / labels.shape[0]

        return loss

    
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_and_evaluate(csv_path, epochs, learning_rate, 
                    K_size, output_size, batch_size=32):
    """
    Train the CNN on all the training data, evaluate on the validation data, 
    and plot the accuracy and loss curves.
    
    Args:
    - csv_path (str): Path to the CSV file containing the dataset.
    - epochs (int): Number of epochs to train the model.
    - learning_rate (float): Learning rate for the optimizer.
    - K_size (int): Size of the convolution kernel.
    - output_size (int): Number of output classes.
    - batch_size (int): Number of samples per batch.
    """
    # Load and preprocess the data
    images, labels = load_and_preprocess_data(csv_path, "Training")

    # images = images[:1]
    # labels = labels[:1]  # TODO REMOVE AFTER DEVELOPING

    input_channels = 1  # grayscale
    fc_input_size = 48 * 48  # Example: assuming 48x48 input image and 32 filters in the first layer
    filters_per_layer = [128, 256, 512, 512]
    
    # Initialize filters, biases, and fully connected layers
    filters_list = []
    biases_list = []
    gamma_list = []
    beta_list = []

    for num_filters in filters_per_layer:
        filters = np.random.normal(0, np.sqrt(2. / (K_size * K_size * input_channels)),
                            (num_filters, K_size, K_size, input_channels))  # shape: (out_c, k_h, k_w, in_c)

        bias = np.zeros(num_filters)
        gamma = np.ones((num_filters,))
        beta = np.zeros((num_filters,))

        filters_list.append(filters)
        biases_list.append(bias)
        gamma_list.append(gamma)
        beta_list.append(beta)

        input_channels = num_filters  # output channels become input for next layer
        # CONV layer output
    output_shape = (batch_size,  filters_per_layer[-1], K_size, K_size)

    # Fully connected layers weights and biases (as a list with 2 entries)
    fc_input_size = output_shape[1] * output_shape[2] * output_shape[3]

    # Now initialize the FC weights with the correct input size
    fc_weights = [
        np.random.normal(0, np.sqrt(2. / fc_input_size), (fc_input_size, 512)),
        np.random.normal(0, np.sqrt(2. / 512), (512, 256))
    ]
    fc_bias = [
        np.zeros(512),
        np.zeros(256)
    ]
    
    # Output layer weights and biases
    output_weights = np.random.normal(0, np.sqrt(2. / 256), (256, output_size))
    output_bias = np.zeros(output_size)
    
    # Initialize the model with gamma_list and beta_list
    machine = EmotionCNN(
        filters_list=filters_list,       # Convolution filters list
        biases_list=biases_list,         # List of biases for each convolution layer
        fc_weights=fc_weights,           # List of fully connected layer weights
        fc_bias=fc_bias,                 # List of biases for fully connected layers
        output_weights=output_weights,   # Output layer weights
        output_bias=output_bias,         # Output layer bias
        gamma=gamma_list,                # List of gamma for batch normalization
        beta=beta_list,                  # List of beta for batch normalization
        step=1,                          # Stride for convolution
        pool_size=2,                     # Pooling window size
        pool_step=2,                     # Stride for pooling
        dropout_rate_conv=0.4,           # Dropout rate after conv layers
        dropout_rate_fc=0.3              # Dropout rate after fully connected layers
    )

    # Create mini-batches for training
    def create_batches(data, labels, batch_size):
        """ Helper function to create batches """
        indices = np.random.permutation(len(data))
        for i in range(0, len(data), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield data[batch_indices], labels[batch_indices]

    # Training the model on all of the training data
    train_losses = []
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        epoch_train_loss = 0
        # Create batches from the training data
        num_batches = 0

        for batch_images, batch_labels in create_batches(images, labels, batch_size):
            num_batches += 1

            # print("batch started")
            # Forward pass for the entire batch
            batch_preds = machine.forward(batch_images)  # Forward the entire batch
            # print("forw happened")
            # Compute loss for the batch
            batch_losses = [cross_entropy_loss(preds, one_hot) 
                            for preds, label in zip(batch_preds, batch_labels) 
                            for one_hot in [np.eye(output_size)[label]]]
            batch_loss = np.mean(batch_losses)
            epoch_train_loss += batch_loss
            # print("sad")
            # Backward pass for the batch
            machine.backward(batch_images, batch_labels, learning_rate)
            # print("back happened")

        avg_train_loss = epoch_train_loss / num_batches
        print(f"loss for this epoch: {avg_train_loss}")
        train_losses.append(avg_train_loss)
    
    # Load the validation data
    val_images, val_labels = load_and_preprocess_data(csv_path, "PrivateTest")
    # val_images = val_images[:1]
    # val_labels = val_labels[:1] # TODO remove after developing
    # Evaluate the model on the validation data
    val_probs = machine.forward(val_images)
    val_preds = np.argmax(val_probs, axis=1)
    accuracy = accuracy_score(np.argmax(val_labels, axis=1), val_preds)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # Plot the training loss and validation accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epochs")
    plt.legend()
    plt.figure()
    plt.plot(range(1, epochs + 1), [accuracy * 100] * epochs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy vs. Epochs")
    plt.legend()
    plt.show()

    # Save the model (optional)
    hyperparams = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "K_size": K_size,
        "num_layers": len(filters_per_layer),
        "output_size": output_size
    }

    save_model(machine, hyperparams, "trained_model.npz")  

def main():
    # Get the absolute path to the CSV file based on the script's location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, '..', '..', 'resources', 'faces')
    csv_path = os.path.join(data_folder, 'data.csv')
    # Train and evaluate the model using all the data
    train_and_evaluate(csv_path, epochs=10, learning_rate=0.0005, 
                    K_size=3, output_size=7,batch_size=32)
if __name__ == "__main__":
    main()

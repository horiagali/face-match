import numpy as np
from model import EmotionCNN
from Data_loader import load_and_preprocess_data, split_data
from sklearn.metrics import accuracy_score

def cross_entropy_loss(predictions, labels):
    m = labels.shape[0]
    loss = -np.sum(labels * np.log(predictions)) / m
    return loss

def sgd_optimizer(model, X_train, y_train, learning_rate=0.01, epochs=10):
    for epoch in range(epochs):
        for i in range(len(X_train)):
            # Forward pass
            output = model.forward(X_train[i])

            # Compute loss
            loss = cross_entropy_loss(output, y_train[i])

            # Backpropagation (you'll need to implement this part)
            # Update the filters using the gradients calculated during backpropagation

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

def main():
    csv_path = 'path_to_your_csv_file.csv'
    images, labels = load_and_preprocess_data(csv_path)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(images, labels)

    # Initialize the model
    K_size = 3
    num_layers = 3
    filters_list = [np.random.randn(K_size, K_size) for _ in range(num_layers)]
    model = EmotionCNN(filters_list, output_size=8)

    # Train the model
    sgd_optimizer(model, X_train, y_train)

    # Evaluate the model
    predictions = []
    for img in X_test:
        output = model.forward(img)
        predictions.append(np.argmax(output))

    accuracy = accuracy_score(np.argmax(y_test, axis=1), predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

import numpy as np
from perceptron import Perceptron
from dataset import load_images
import torch

def main():
    # Load your dataset
    X_train, y_train = load_images('images/train')  # Adjust path accordingly
    X_test, _ = load_images('images/test')  # Assuming test labels are not needed for prediction

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)  # Convert input data to tensor
    y_train_tensor = torch.LongTensor(y_train)  # Convert labels to tensor (CrossEntropy expects LongTensor for labels)

    input_size = X_train.shape[1]  # Should be 400 (flattened 20x20 pixels)
    num_classes = 10  # Digits 0-9

    # Initialize Perceptron (now an MLP) with hidden layers and ReLU activation
    perceptron = Perceptron(input_size=input_size, num_classes=num_classes, hidden_layers=[64, 32], dropout_rate=0.3, activation_fn='relu')

    # Train the Perceptron (MLP) with adjusted parameters
    perceptron.fit(X_train_tensor, y_train_tensor, learning_rate=0.0001, epochs=2000)

    # Convert X_test to a PyTorch FloatTensor and ensure it's flattened
    X_test = np.array([x.flatten() for x in X_test])  # Ensure consistent shape
    X_test_tensor = torch.FloatTensor(X_test)

    # Test the trained Perceptron
    predictions = perceptron.predict(X_test_tensor)

    # Print results
    print("Training Complete.")
    print(f"Predictions on test images: {predictions}")

if __name__ == '__main__':
    main()

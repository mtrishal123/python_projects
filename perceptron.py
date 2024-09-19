import torch
import torch.nn as nn
import torch.optim as optim

class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=None, dropout_rate=0.0, activation_fn='relu'):
        super(Perceptron, self).__init__()

        # List of layers
        layers = []

        # Add hidden layers
        in_features = input_size
        if hidden_layers:
            for hidden_units in hidden_layers:
                layers.append(nn.Linear(in_features, hidden_units))
                if activation_fn == 'relu':
                    layers.append(nn.ReLU())
                elif activation_fn == 'sigmoid':
                    layers.append(nn.Sigmoid())
                layers.append(nn.Dropout(dropout_rate))
                in_features = hidden_units

        # Add the output layer
        layers.append(nn.Linear(in_features, num_classes))

        # Define the sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def fit(self, X_train, y_train, learning_rate=0.001, epochs=1000):
        self.train()  # Set the model to training mode
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X_test):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(X_test)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

import numpy as np
import matplotlib.pyplot as plt

# Define ReLU function
def relu(x):
    return np.maximum(x, 0)

# Compute loss and gradient for a single sample
def compute_loss_and_gradient(w, X, y):
    predictions = relu(np.dot(X, w))
    error = y - predictions
    loss = np.mean(error ** 2)  # Mean squared error

    # Gradient computation
    # gradient = -2 * np.dot(X.T, error * (np.dot(X, w) > 0)) / len(y)
    gradient = -2 * np.dot(X.T, error * (np.dot(X, w) > 0)) 
    
    return loss, gradient

# SGD with replacement
def SGD_with_replacement(X, y, learning_rate, epochs):
    d = X.shape[1]  # Number of features
    w = np.random.randn(d)  # Initialize weights
    N = len(y)  # Number of samples
    average_losses = []  # List to store the average loss of each epoch

    for epoch in range(epochs):
        epoch_loss = 0  # Accumulate the loss for the current epoch

        # Perform N updates, each with a randomly selected data point (with replacement)
        for _ in range(N):
            i = np.random.randint(N)  # Randomly choose an index
            xi = X[i, :]
            yi = y[i]
            xi = xi.reshape(1, -1)  # Reshape to match the dimensions
            
            # Compute loss and gradient for the randomly selected sample
            loss, grad = compute_loss_and_gradient(w, xi, yi)
            
            # Update weights
            w -= learning_rate * grad
            
            # Accumulate the loss for this sample
            epoch_loss += loss

        # Compute the average loss for the current epoch
        average_epoch_loss = epoch_loss / N
        average_losses.append(average_epoch_loss)  # Save the average loss

        # Print the average loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, MSE Loss: {average_epoch_loss}')

    return w, average_losses

# Simulate some data
np.random.seed(22)
d = 200  # Number of features
N = 1000  # Number of samples
X = np.random.randn(N, d)
true_w = np.random.randn(d)
y = relu(np.dot(X, true_w)) + np.random.normal(0, 0.05, N)

# Run SGD
learning_rate = 0.001
epochs = 1000
w_learned, average_losses = SGD_with_replacement(X, y, learning_rate, epochs)

# Plot the average loss per epoch
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), np.log(average_losses), label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('log MSE Loss')
plt.title('MSG Loss vs Epoch using SGD with replacement')
plt.legend()
plt.grid(True)
plt.savefig('p5_a.png')


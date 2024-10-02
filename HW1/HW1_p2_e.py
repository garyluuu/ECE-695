import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
d = 10
N = 100
sigma = 0.01
T = 2000
learning_rate = 0.01  

# Generate data
np.random.seed(22)
X = np.random.randn(N, d)
w_star = np.random.randn(d, 1)
noise = sigma * np.random.randn(N, 1)
y = X @ w_star + noise

# Initialize weights
w = np.zeros((d, 1))

# SGD
loss_history = []

for t in range(T):
    i = np.random.randint(N)  # Randomly pick one data point
    X_i = X[i:i+1, :]
    y_i = y[i:i+1, :]
    
    # Compute prediction and loss
    y_pred = X_i @ w
    loss = 0.5 * np.mean((y_i - y_pred) ** 2)
    loss_history.append(loss)
    
    # Compute gradient
    gradient = X_i.T @ (y_pred - y_i)
    
    # Update weights
    w -= learning_rate * gradient
    
# Plot loss function
plt.plot(range(T), loss_history)
# plt.plot(range(T), np.log(loss_history))
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Linear Regression Loss Function vs. Iteration')
# plt.show()

script_name = os.path.basename(__file__)  # Get the script name with extension
base_name, _ = os.path.splitext(script_name)  # Remove the file extension
plt.savefig(f'{base_name}_loss')

# Report the learning rate
print(f"Learning rate used: {learning_rate}")

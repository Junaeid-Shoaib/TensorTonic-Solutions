import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    # 1. Initialize parameters
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    b = 0.0

    for i in range(steps):
        # 2. Linear combination + Activation
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 3. Calculate gradients
        error = p - y
        grad_w = (X.T @ error) / num_samples
        grad_b = np.mean(error)
        
        # 4. Update parameters
        w -= lr * grad_w
        b -= lr * grad_b
        
    return w, b

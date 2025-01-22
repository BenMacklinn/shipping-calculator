from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Input data
X = np.array([8148, 3829, 8064, 5763, 5500])
y = np.array([207.45, 104.28, 203.95, 142.93, 148.53])

# Feature scaling
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std

# Initialize parameters
w = 0
b = 0
learning_rate = 0.01
num_iterations = 5000

def compute_cost(X, y, w, b):
    m = len(X)
    cost = (1 / (2 * m)) * np.sum((w * X + b - y) ** 2)
    return cost

def gradient_descent(X, y, w, b, learning_rate, iterations):
    m = len(X)
    for i in range(iterations):
        dw = (1 / m) * np.sum((w * X + b - y) * X)
        db = (1 / m) * np.sum(w * X + b - y)
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b

# Perform gradient descent
w, b = gradient_descent(X_scaled, y, w, b, learning_rate, num_iterations)
w_unscaled = w / X_std
b_unscaled = b - w_unscaled * X_mean

@app.route('/')
def index():
    # Pass the parameters to the template
    return render_template('index.html', w_unscaled=w_unscaled, b_unscaled=b_unscaled)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

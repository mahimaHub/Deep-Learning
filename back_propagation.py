# -*- coding: utf-8 -*-
"""Back Propagation

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G9sGLV97MTc3p-kp8Wf8SHkKF9QNGsKN
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  #TfidfVectorizer → Converts text into numerical form using TF-IDF
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)   #np.random.randn() initializes weights with random values from a normal distribution
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))   #The sigmoid function squashes values into a range between 0 and 1

    def sigmoid_derivative(self, x):
        return x * (1 - x)            #derivative of the sigmoid function

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1   #Computes the weighted sum of inputs for the hidden layer.
        self.a1 = self.sigmoid(self.z1)      #Applies the sigmoid activation function
        self.z2 = np.dot(self.a1, self.W2) + self.b2    #Computes the weighted sum for the output layer
        self.a2 = self.sigmoid(self.z2)             #Applies sigmoid activation again to get final output
        return self.a2

    def backward(self, X, y):
        m = y.shape[0]

        # Compute error
        error = self.a2 - y           #Difference between predicted and actual values
        d_output = error * self.sigmoid_derivative(self.a2)   #Computes the gradient of the loss with respect to the outpu

        # Compute gradient for hidden layer
        error_hidden = np.dot(d_output, self.W2.T)          #Propagates the error backward from the output layer to the hidden layer
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)    #Computes the gradient for the hidden layer

        # Update weights and biases
        self.W2 -= self.learning_rate * np.dot(self.a1.T, d_output) / m
        self.b2 -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True) / m
        self.W1 -= self.learning_rate * np.dot(X.T, d_hidden) / m
        self.b1 -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True) / m

    def train(self, X, y, epochs=500):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

            if epoch % 100 == 0:
                loss = np.mean((self.a2 - y) ** 2)                    #Runs forward and backward propagation for multiple epochs
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)

# Load dataset
file_path = "/content/spam.csv"  # Update this path if needed
df = pd.read_csv(file_path, encoding='latin-1')

# Keep only necessary columns
df = df[['Category', 'Message']]
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})                    #Converts categorical labels into numerical values

# Convert text to numerical format
vectorizer = TfidfVectorizer(max_features=1000)         #TfidfVectorizer(max_features=1000) → Creates a TF-IDF vectorizer with a maximum of 1000 words.
vectorizer.fit_transform(df['Message'])                 #Converts the messages into a numerical representation
X = vectorizer.fit_transform(df['Message']).toarray()
y = df['Category'].values.reshape(-1, 1)  # Reshape for compatibility

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the MLP
mlp = MLP(input_size=1000, hidden_size=256, output_size=1, learning_rate=0.01)
mlp.train(X_train, y_train, epochs=500)

# Predictions
y_pred = mlp.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
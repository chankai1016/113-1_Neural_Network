import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-Hot encode tags
encoder = OneHotEncoder(sparse_output=False)  # Updated argument
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the data set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a neural network-like model
model = Sequential([
    Dense(10, input_dim=4, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.1)

# Evaluation model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"測試集損失: {loss}")
print(f"測試集準確率: {accuracy}")

# Forecasting example
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Iris-setosa 範例
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)
print(f"預測類別: {iris.target_names[predicted_class]}")
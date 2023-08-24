import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Load the data from measurements.txt and mediapipe_data.txt
with open('data/measurements.txt', 'r') as file:
    measurements = np.loadtxt(file)

with open('data/mediapipe_data.txt', 'r') as file:
    mediapipe_data = np.loadtxt(file)

# Prepare the input and target variables
X = mediapipe_data.reshape(-1, 4, 1)  # Reshape mediapipe data into (num_samples, 4, 1)
y = measurements.reshape(-1, 4)  # Reshape measurements into (num_samples, 4)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Save the mean and scale values
mean_values = scaler.mean_
scale_values = scaler.scale_

with open('scaler_values.pkl', 'wb') as file:
    pickle.dump((mean_values, scale_values), file)

# Build the model
model = keras.Sequential([
    layers.Flatten(input_shape=(4, 1)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error:", mse)

# Save the trained model
model.save('trained_model.h5')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
import joblib

# Load the dataset
data = pd.read_csv("Student_Performance.csv")

# Convert 'Yes'/'No' to 1/0 for 'Extracurricular Activities'
data['Extracurricular Activities'] = data['Extracurricular Activities'].replace({'Yes': 1, 'No': 0})

# Preprocess the data
features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split data into training and testing sets
X = data[features]
y = data['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

# Save the trained model in Keras format
model.save('model.keras')
print("Model and scaler saved successfully.")

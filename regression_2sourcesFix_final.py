print('----------- regression MLP for two sources with fixed strength -------------------')
print('----------- Copyright (C) 2024 N. Pallikarakis ----------------------------------')

import tensorflow as tf  # Importing TensorFlow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Set random seed for reproducibility
np.random.seed(100)
tf.random.set_seed(42)

# Load dataset
train_data = np.loadtxt('2s_fixed_strength_test.csv', delimiter=',', skiprows=1)

# Shuffle data
np.random.shuffle(train_data)

# Define X (features) and Y (targets)
n = train_data.shape[1]
X_tr = train_data[:, 7:n+1]
column_indices = [1, 2, 4, 5]
Y_tr = train_data[:, column_indices]

# Handle negative angles
Y_tr[Y_tr < 0] += 2 * np.pi

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, Y_tr, test_size=0.3, random_state=10)

# Standardize features and targets
scaler_X = StandardScaler()
X_train_ss = scaler_X.fit_transform(X_train)
X_val_ss = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train_ss = scaler_y.fit_transform(y_train)
y_val_ss = scaler_y.transform(y_val)

# Build the model
model = Sequential([
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dense(4, activation=None)
])

# Compile the model with Adam optimizer
opt = Adam(learning_rate=0.00005)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[RootMeanSquaredError()])

# Early stopping callback
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=150, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(X_train_ss, y_train_ss, epochs=2000, batch_size=64, validation_data=(X_val_ss, y_val_ss), callbacks=[monitor])

# Predictions on training and validation sets
Y_predict_tr = model.predict(X_train_ss)
Y_predict_val = model.predict(X_val_ss)

# Inverse transform predictions and targets
y_pred_inv = scaler_y.inverse_transform(Y_predict_val)
y_val_inv = scaler_y.inverse_transform(y_val_ss)

# R-squared scores
print('R squared training set:', round(r2_score(y_train_ss, Y_predict_tr) * 100, 2))
print('R squared validation set:', round(r2_score(y_val_ss, Y_predict_val) * 100, 2))

# Individual R-squared scores
print('R squared real Dipole distance:', round(r2_score(y_val_inv[:, 0], y_pred_inv[:, 0]) * 100, 2))
print('R squared real Dipole angle:', round(r2_score(y_val_inv[:, 1], y_pred_inv[:, 1]) * 100, 2))
print('R squared real Dipole distance 2:', round(r2_score(y_val_inv[:, 2], y_pred_inv[:, 2]) * 100, 2))
print('R squared real Dipole angle 2:', round(r2_score(y_val_inv[:, 3], y_pred_inv[:, 3]) * 100, 2))

# RMSE for validation set
print('RMSE:', round(np.sqrt(mean_squared_error(y_val, Y_predict_val)), 3))

# Model summary
print(model.summary())

# Plotting the training history
plt.plot(history.history['root_mean_squared_error'], label='train_rmse')
plt.plot(history.history['val_root_mean_squared_error'], label='val_rmse')
plt.title('Model RMSE')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Scatter plot function
def plot_scatter(true_vals, pred_vals, feature_name, color):
    plt.scatter(true_vals, pred_vals, label=f'Predicted {feature_name}', marker='o', facecolors='none', edgecolors=color)
    m, b = np.polyfit(true_vals, pred_vals, 1)
    plt.plot(true_vals, m * np.array(true_vals) + b, label=f'Regression Line (y = {m:.2f}x + {b:.2f})', linestyle='--', color='black')
    plt.xlabel('True values', fontsize=14)
    plt.ylabel('Predicted values', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

# Scatter plots for predictions vs actual values
plot_scatter(y_val_inv[:, 0], y_pred_inv[:, 0], 'Distance 1', 'blue')
plot_scatter(y_val_inv[:, 1], y_pred_inv[:, 1], 'Angle 1', 'red')
plot_scatter(y_val_inv[:, 2], y_pred_inv[:, 2], 'Distance 2', 'blue')
plot_scatter(y_val_inv[:, 3], y_pred_inv[:, 3], 'Angle 2', 'red')

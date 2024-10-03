###### Nikolaos Pallikarakis ###########

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Set random seed for reproducibility
np.random.seed(100)
tf.random.set_seed(42)

# Load the dataset
train_data = np.loadtxt('2s_noisy_strength_10_test.csv', delimiter=',', skiprows=1)

# Shuffle the dataset
np.random.shuffle(train_data)

# Define the input features (X_tr) and target (Y_tr)
n = train_data.shape[1]
X_tr = train_data[:, 7:n+1]
Y_tr = train_data[:, 1:7]

# Adjust negative angles by adding 2π
Y_tr[Y_tr < 0] += 2 * np.pi

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, Y_tr, test_size=0.3, random_state=10)

# Standardize features and targets
scaler_X = StandardScaler()
X_train_ss = scaler_X.fit_transform(X_train)
X_val_ss = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train_ss = scaler_y.fit_transform(y_train)
y_val_ss = scaler_y.transform(y_val)

# Define the model
model = Sequential([
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dropout(0.2, seed=42),
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dropout(0.2, seed=42),
    Dense(500, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dropout(0.2, seed=42),
    Dense(6, activation=None)
])

# Compile the model
opt = Adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[RootMeanSquaredError()])

# Early stopping callback
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=100, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(X_train_ss, y_train_ss, epochs=2500, batch_size=64, validation_data=(X_val_ss, y_val_ss), callbacks=[monitor])

# Predictions
Y_predict_tr = model.predict(X_train_ss)
Y_predict_val = model.predict(X_val_ss)

# Inverse transform predictions back to original scale
y_pred_inv = scaler_y.inverse_transform(Y_predict_val)
y_val_inv = scaler_y.inverse_transform(y_val_ss)

# R-squared scores for training and validation sets
print('R squared training set:', round(r2_score(y_train_ss, Y_predict_tr) * 100, 2))
print('R squared validation set:', round(r2_score(y_val_ss, Y_predict_val) * 100, 2))

# Individual R-squared scores for each target feature
for i in range(6):
    print(f'R squared for feature {i+1}:', round(r2_score(y_val_inv[:, i], y_pred_inv[:, i]) * 100, 2))

# RMSE for validation set
print('RMSE validation set:', round(np.sqrt(mean_squared_error(y_val, Y_predict_val)), 3))

# Model summary
print(model.summary())

# Plot training history
def plot_history(history, metric_name, title):
    plt.plot(history.history[metric_name], label='train')
    plt.plot(history.history[f'val_{metric_name}'], label='validate')
    plt.title(title)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.show()

plot_history(history, 'root_mean_squared_error', 'Model Accuracy')
plot_history(history, 'loss', 'Model Loss')

# Scatter plot function
def plot_scatter(true_vals, pred_vals, feature_name, color):
    plt.scatter(true_vals, pred_vals, label=f'Predicted {feature_name}', marker='o', facecolors='none', edgecolors=color)
    m, b = np.polyfit(true_vals, pred_vals, 1)
    plt.plot(true_vals, m * np.array(true_vals) + b, label=f'Regression Line (y = {m:.2f}x + {b:.2f})', linestyle='--', color='black')
    plt.xlabel('True values', fontsize=14)
    plt.ylabel('Predicted values', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

# Plot scatter plots for each feature
feature_names = ['Distance 1', 'Angle 1', 'Strength 1', 'Distance 2', 'Angle 2', 'Strength 2']
colors = ['blue', 'red', 'orange', 'blue', 'red', 'orange']
for i in range(6):
    plot_scatter(y_val_inv[:, i], y_pred_inv[:, i], feature_names[i], colors[i])

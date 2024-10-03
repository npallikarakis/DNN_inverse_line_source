print('----------- regression MLP for one source -------------------')
print('----------- Copyright (C) 2024 N. Pallikarakis ----------------------------------')


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error

# Set random seed for reproducibility
np.random.seed(10)
tf.random.set_seed(42)

# Define the model
model = Sequential([
    Dense(50, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dense(100, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-4)),
    Dense(3, activation=None)
])

# Compile the model with Adam optimizer
opt = Adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[RootMeanSquaredError()])

# Early stopping callback
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, verbose=1, restore_best_weights=True)

# Load and shuffle data
df = pd.read_csv("fullfields_1_test.csv", na_values=['NA', '?'])
df = df.reindex(np.random.permutation(df.index))

# Split the data into input (X_tr) and target (Y_tr)
n = len(df.columns)
X_tr = df.iloc[1:10001, 4:n+1].values  # Features
Y_tr = df.iloc[1:10001, 1:4].values    # Targets

# Train and validation split
X_train, X_val, y_train, y_val = train_test_split(X_tr, Y_tr, test_size=0.3, random_state=10)

# Standardize the features and targets
scaler_X = StandardScaler()
X_train_ss = scaler_X.fit_transform(X_train)
X_val_ss = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train_ss = scaler_y.fit_transform(y_train)
y_val_ss = scaler_y.transform(y_val)

# Train the model
history = model.fit(X_train_ss, y_train_ss, validation_data=(X_val_ss, y_val_ss), verbose=2, callbacks=[monitor],epochs=2000)

# Predict on training and validation sets
Y_predict_tr = model.predict(X_train_ss)
Y_predict_val = model.predict(X_val_ss)

# Inverse transform predictions to original scale
y_pred_inv = scaler_y.inverse_transform(Y_predict_val)
y_val_inv = scaler_y.inverse_transform(y_val_ss)

# Print R-squared scores
print('R squared training set:', round(r2_score(y_train_ss, Y_predict_tr) * 100, 2))
print('R squared validation set:', round(r2_score(y_val_ss, Y_predict_val) * 100, 2))

# Print R-squared scores for individual features
print('R squared Dipole Distance:', round(r2_score(y_val_inv[:, 0], y_pred_inv[:, 0]) * 100, 2))
print('R squared Dipole Angle:', round(r2_score(y_val_inv[:, 1], y_pred_inv[:, 1]) * 100, 1))
print('R squared Dipole Strength:', round(r2_score(y_val_inv[:, 2], y_pred_inv[:, 2]) * 100, 2))

# Print RMSE for validation set
print('RMSE validation set:', round(np.sqrt(mean_squared_error(y_val_ss, Y_predict_val)), 3))

# Scatter plot for validation vs predicted
def plot_scatter(true_vals, pred_vals, feature_name, color):
    plt.scatter(true_vals, pred_vals, label=f'Predicted {feature_name}', marker='o', facecolors='none', edgecolors=color)
    m, b = np.polyfit(true_vals, pred_vals, 1)
    plt.plot(true_vals, m * np.array(true_vals) + b, label=f'Regression Line (y = {m:.2f}x + {b:.2f})', linestyle='--', color='black')
    plt.xlabel('True values', fontsize=14)
    plt.ylabel('Predicted values', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

plot_scatter(y_val_inv[:, 0], y_pred_inv[:, 0], 'Distance', 'blue')
plot_scatter(y_val_inv[:, 1], y_pred_inv[:, 1], 'Angle', 'red')
plot_scatter(y_val_inv[:, 2], y_pred_inv[:, 2], 'Strength', 'orange')

# Summarize training history
plt.plot(history.history['root_mean_squared_error'], label='train_rmse')
plt.plot(history.history['val_root_mean_squared_error'], label='val_rmse')
plt.title('Model Accuracy')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Print the model summary
print(model.summary())

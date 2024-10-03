####Nikolaos Pallikarakis##############

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Display TensorFlow version
print("TensorFlow version:", tf.__version__)

# Set the random seed for reproducibility
tf.random.set_seed(42)

# Load dataset
df = pd.read_csv("2class_test.csv", na_values=['NA', '?'])
n = len(df.columns)
X1 = df.iloc[0:160000, 1:n+1].values
y1 = df.iloc[0:160000, 0].values - 1  # Map to 0-based indexing

# Define the model
model = Sequential([
    Dense(50, input_dim=X1.shape[1], activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-3)),
    Dense(100, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-3)),
    Dense(100, activation=LeakyReLU(alpha=0.01), activity_regularizer=regularizers.l2(2e-3)),
    Dense(2, activation='softmax')
])

# Compile the model with Adam optimizer
opt = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# One-hot encode labels
Y_tr = to_categorical(y1, num_classes=2)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X1, Y_tr, test_size=0.3, random_state=1000)

# Standardize features
scaler = StandardScaler()
X_train_ss = scaler.fit_transform(X_train)
X_val_ss = scaler.transform(X_val)

# Define early stopping callback
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(X_train_ss, y_train, batch_size=64, validation_data=(X_val_ss, y_val), callbacks=[monitor], verbose=2, epochs=800)

# Training results
Y_predict_tr = model.predict(X_train_ss)
Y_predict_tr2 = np.argmax(Y_predict_tr, axis = 1)
y_tr2=np.argmax(y_train, axis = 1)

# Validating results
Y_predict_val = model.predict(X_val_ss)
Y_predict_val2 = np.argmax(Y_predict_val, axis = 1)
y_val2=np.argmax(y_val, axis = 1)

acc_mlp = accuracy_score(y_val2, Y_predict_val2)
print(f'Validation accuracy: {acc_mlp}')


############ updated confucion matrix ########
# Create confusion matrix
cmp = confusion_matrix(y_val2, Y_predict_val2)
cmp = cmp.astype('float') / cmp.sum(axis=1)[:, np.newaxis]
# Plot confusion matrix with gridlines
fig, ax = plt.subplots(figsize=(8, 5))
cmp_display = ConfusionMatrixDisplay(cmp, display_labels=["1 source", "2 sources"])
# Plot confusion matrix without changing the color of numbers inside the boxes
cmp_display.plot(ax=ax, cmap='Blues', values_format=".2f") #YlOrRd, Purples, Blues
# Add gridlines to separate each box
for i in range(len(cmp) - 1):
    ax.axhline(i + 0.5, color='black', linewidth=1)
for j in range(len(cmp[0]) - 1):
    ax.axvline(j + 0.5, color='black', linewidth=1)
# Show the plot

# Set font size for title, axis labels, and tick labels
ax.set_title('Confusion Matrix', fontsize=16,fontweight='bold')  # Change 16 to your desired font size
ax.set_xlabel('predicted labels', fontsize=14,fontweight='bold')  # Change 14 to your desired font size
ax.set_ylabel('true labels', fontsize=14,fontweight='bold')  # Change 14 to your desired font size

# Set font size for tick labels
ax.tick_params(axis='both', which='both', labelsize=12)  # Change 12 to your desired tick label font size

# Set face color to white (opaque) for all elements
fig.patch.set_facecolor('white')
ax.patch.set_facecolor('white')

plt.show()


# Print Validation classification report
target_names = ["source#1", "source#2"]
print('Validation Classification Report:')
print(classification_report(y_val2, Y_predict_val2, target_names=target_names,digits=4))

# Print Training classification report
print('Training Classification Report:')
target_names = ["source#1", "source#2"]
print(classification_report(y_tr2, Y_predict_tr2,target_names=target_names,digits=4))

# Plot model loss over epochs
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

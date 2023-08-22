import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Configuration parameters
BATCH_SIZE = 50
IMG_DIMS = (32, 32, 3)
NUM_CLASSES = 100
EPOCHS = 10
VALIDATION_RATIO = 0.2
VERBOSE_MODE = 1
LOSS = sparse_categorical_crossentropy
OPTIMIZER = Adam() #Adam performs better than RMS so far

# Get CIFAR-100 dataset
(train_imgs, train_labels), (test_imgs, test_labels) = cifar100.load_data()

# Convert datatype and normalize
train_imgs = train_imgs.astype('float32') / 255.0
test_imgs = test_imgs.astype('float32') / 255.0

# Building the Neural Network
network = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_DIMS),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compilation step
network.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

# Train the model
train_history = network.fit(
    train_imgs, train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_RATIO,
    verbose=VERBOSE_MODE
)

# Evaluate the model
test_loss, test_acc = network.evaluate(test_imgs, test_labels, verbose=0)
print(f"Testing Loss: {test_loss} | Testing Accuracy: {test_acc}")

# Plotting the results
# Loss Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_history.history['val_loss'])
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(train_history.history['val_accuracy'])
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.tight_layout()
plt.show()

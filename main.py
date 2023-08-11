import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

(train_data, train_labels), (test_data, test_labels) = cifar100.load_data()

train_data = train_data/255
test_data = test_data/255




model.compile(optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
    train_data, train_labels,
    epochs=6,
    batch_size=512,
    validation_split=0.2
)
test_loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy : {accuracy*100:.2f}%")
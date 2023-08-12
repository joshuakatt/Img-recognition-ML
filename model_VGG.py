import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(train_data, train_labels), (test_data, test_labels) = cifar100.load_data()

train_data = train_data/255
test_data = test_data/255

base_model = VGG16(weights='imagenet', include_top=False, input_shape = (32,32,3))

for layer in base_model.layers[:-4]:
    layer.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(100, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs = predictions)

model.compile(optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(train_data)

history = model.fit(datagen.flow(train_data, train_labels, batch_size=64),
    epochs=6,
    validation_data=(test_data, test_labels),
    validation_split = 0.2
)
test_loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy : {accuracy*100:.2f}%")
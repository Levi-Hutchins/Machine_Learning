'''
This program is a simple implementation of machine learning basics using Tensorflow.
While the code in its entirety is not mine, i have manipulated and added some features
to experiment with the model. I have read all the related functions documentation
from tensorflow and have documented what they are doing to gain an understanding
Current accuracy = 61
'''

#The below environment variable simply disables the GPU errors

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import Visualise
import numpy as np
import pandas as pd
print("Current TensorFlow version:", tf.__version__)

# Different sized images must normalise
'''
img1 = Image.open("Dataset/training_set/dog.10.jpg")
print(np.array(img1).shape)
img2 = Image.open("Dataset/training_set/dog.100.jpg")
print(np.array(img2).shape)
'''

training_dataGen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
testing_dataGen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

training_images = training_dataGen.flow_from_directory(
    directory="Dataset/training_set/training_set/",
    target_size=(224,224),
    class_mode="categorical",
    batch_size=80,
    seed=42
)
testing_images = training_dataGen.flow_from_directory(
    directory="Dataset/test_set/test_set/",
    target_size=(224,224),
    class_mode="categorical",
    batch_size=80,
    seed=42
)

group1 = training_images.next()
'''
Validating the images have been transformed / normalised 
group1 = training_images.next()
print(group1[0].shape)
'''
Visualise.visualiseGroup(group1)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3), input_shape=(224,224 ,3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax"),

])

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(),
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
)
print(model.summary())

h = model.fit(
    training_images,
    validation_data=testing_images,
    epochs=5
)






print("... Tests Complete ...")

'''
The below environment variables simply disables the GPU errors
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import Visualise as vs
import numpy as np
import pandas as pd
print("Current TensorFlow version:", tf.__version__)

# Using the dataset from tensor flow mnist which is a collection of handwritten digits
mnist_dataset = tf.keras.datasets.mnist

# Loading the data into their respective training and testing variables
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Verifying the training and testing images are of the same pixel dimensions
print(train_images.shape, test_images.shape)

classifications = [0,1,2,3,4,5,6,7,8,9]


train_images = train_images / 255
test_images = test_images / 255


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(20)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]

)

model.fit(train_images, train_labels, epochs=10)

loss, acc = model.evaluate(test_images, test_labels,verbose=2)
print(loss)
print(acc)


print("... Tests Complete ...")
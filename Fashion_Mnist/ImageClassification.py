'''
The below environment variables simply disables the GPU errors
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
print("Current TensorFlow version:", tf.__version__)
clothing_dataset = tf.keras.datasets.fashion_mnist
'''
Loading the data returns four numpy arrays.
train_images and train_lables are the training set ( 
the data the models uses to learn)
The model is tested against the test set - test_images & test_labels
'''
(train_images, train_labels), (test_images, test_labels) = clothing_dataset.load_data()

clothes_classification = ["T-Shirt", "Pants", "Top", "Dress", "Coat",
                          "Sandal", "Shirt", "Sneaker", "Bag","Ankle Boot"]
print(len(train_labels), ": Images in training set "
        "each with ", train_images.shape[1], train_images.shape[2],
        "pixel ratio")

print(len(test_labels), ": Images in test set "
        "each with ", test_images.shape[1], test_images.shape[2],
        "pixel ratio")


train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    # .Flatten transforms the data from a 2d 28 x 28 pixel to a 1d array 784 pixels
    tf.keras.layers.Flatten(input_shape=(28,28)),
    # .Dense(128) is a layer that has 128 neurons
    tf.keras.layers.Dense(128, activation="relu"),
    # .Dense(10) returns a logits array of length 10. Each node contains a score that
    # that indicates the current image belongs to one of the 10 classes
    tf.keras.layers.Dense(10)
])

model.compile(
    # This is how the model is updates based on the data it sees and its loss fucntion
    optimizer="adam",
    # Loss measures how accurate the model is during training
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Used to monitor the training and testing steps
    metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test Accuracy: ", test_acc)
print("--------------")
# Softmax layer converts linear output to probabilities
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
# Make predictions on test images


def makePredictions(test_images):
    predictions = probability_model.predict(test_images)

    try:
        file = open("Image Predictions.txt", "w")
    except IOError:
        print("An error has occured")
        makePredictions(test_images)

    # Prediction on the first 10 items in the test dataset
    with file:
        for i in range(len(predictions)):
            item = clothes_classification[np.argmax(predictions[i])]
            file.write("Prediction : " + item +" with %" +
                       str(predictions[i][np.argmax(predictions[i])]) + " certainty\n")



makePredictions(test_images)
print("...Tests Complete...")
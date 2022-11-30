'''
The below environment variables simply disables the GPU errors
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(42)

import numpy as np
def visualiseGroup(group: tf.keras.preprocessing.image.DirectoryIterator):
    n = 15
    rows, cols = 5, 3
    fix, axes = plt.subplots(rows,cols,figsize=(3 * rows, 3 * cols))

    for i in range(n):
        img = np.array(group[0][i] * 255, dtype="uint8")
        ax = axes[i // cols, i % cols]
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
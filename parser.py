import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def save_to_csv(images, labels, csv_filename):
    """
    Flatten the images, combine them with labels, and save to CSV.
    
    - images: NumPy array of shape (num_samples, 28, 28)
    - labels: NumPy array of shape (num_samples,)
    - csv_filename: Name of the output CSV file
    
    The output file will have '28*28 + 1' columns:
      pixel0, pixel1, ..., pixel783, label
    """
    # Flatten each 28x28 image into a 784-dimensional vector
    images_flat = images.reshape(images.shape[0], -1)
    
    # Combine flattened images and labels column-wise
    data = np.column_stack([images_flat, labels])
    
    # Create a DataFrame
    col_names = [f"pixel{i}" for i in range(784)] + ["label"]
    df = pd.DataFrame(data, columns=col_names)
    
    # Write to CSV (no index column)
    df.to_csv(csv_filename, index=False)
    
    print(f"Saved {csv_filename} with shape {df.shape}.")

if __name__ == "__main__":
    # 1) Load the Fashion MNIST dataset from Keras
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # 2) Save the training set to CSV
    save_to_csv(train_images, train_labels, "fashion_mnist_train.csv")
    
    # 3) Save the testing set to CSV
    save_to_csv(test_images, test_labels, "fashion_mnist_test.csv")

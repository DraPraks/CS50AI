import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Return tuple `(images, labels)`. `images` is a list of all
    images in the data directory, each formatted as a numpy ndarray
    with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` is a list of
    integer labels, representing the categories for each corresponding `image`.
    """

    images = []
    labels = []

    # Loop over each subdirectory in the directory
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        # Check if the path is a directory
        if os.path.isdir(folder_path):

            # Loop over each file in the subdirectory
            for img_file in os.listdir(folder_path):
                # Load the image file
                img = cv2.imread(os.path.join(folder_path, img_file))

                # Resize the image
                resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

                # Add the resized image and its label to the respective lists
                images.append(resized_img)
                labels.append(int(folder))

    return images, labels




def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Initialize a sequential model
    model = tf.keras.models.Sequential()

    # Add a convolutional layer with 32 filters using a 3x3 kernel
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Add a max-pooling layer with a 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Add another convolutional layer with 64 filters using a 3x3 kernel
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

    # Add batch normalization layer
    model.add(tf.keras.layers.BatchNormalization())

    # Add another max-pooling layer with a 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the tensor output from the previous layer
    model.add(tf.keras.layers.Flatten())

    # Add a dense hidden layer with 256 units and ReLU activation
    model.add(tf.keras.layers.Dense(256, activation="relu"))

    # Add dropout to prevent overfitting
    model.add(tf.keras.layers.Dropout(0.6))

    # Add an output layer with `NUM_CATEGORIES` units and softmax activation
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Compile the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Print a summary of the model
    model.summary()

    return model



if __name__ == "__main__":
    main()

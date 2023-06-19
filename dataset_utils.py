import os
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DatasetUtils:
    @staticmethod
    def create_dataset(dir_path: str) -> tuple:
        """
        Create a dataset from images in a directory.

        Args:
            dir_path (str): Path to the directory containing subdirectories with images.

        Returns:
            tuple: A tuple containing two arrays: (data, labels).
                   The data array contains preprocessed image arrays, and the labels array contains the corresponding labels.
        """
        data, labels = [], []
        label = 0
        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        # Load and preprocess the image
                        image = load_img(file_path, grayscale=False, color_mode='rgb', target_size=(40, 40))
                        image = img_to_array(image) / 255.0

                        # Append the preprocessed image array and corresponding label to the lists
                        data.append(image)
                        labels.append(label)

                label += 1

        # Convert the lists to NumPy arrays for improved efficiency
        data = np.array(data)
        labels = np.array(labels)

        return data, labels

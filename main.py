import os
from dataset_utils import DatasetUtils
from model_utils import ModelUtils

# Set the path to the data directory.
train_dir = 'C:/Users/DhrCS/Documents/10_Kaggle/ASL_Dataset/Data/Train'
test_dir = 'C:/Users/DhrCS/Documents/10_Kaggle/ASL_Dataset/Data/Test'

# saved_model_path = "models/ASL_model_20230617_185330.h5" # all data
saved_model_path = ""

# Create an instance of the DatasetUtils class
dataset_utils = DatasetUtils()

# Load and preprocess the dataset, creating the data and label arrays.
print("Loading and preprocessing the dataset...")
data_train, labels0 = dataset_utils.create_dataset(train_dir)
data_test, testlabels0 = dataset_utils.create_dataset(test_dir)

# Create an instance of the ModelUtils class
model_utils = ModelUtils()

# Check if a saved model exists
if os.path.exists(saved_model_path):
    print("Saved model found. Loading the model...")
    model = model_utils.load_saved_model(saved_model_path)
else:
    print("Saved model not found. Training a new model...")
    # Train the model
    model, history, train_data, test_data, train_x, test_x, train_y, test_y = model_utils.train_model(data_train, labels0)
    print("Model training completed.")

    # Evaluate the model on the test dataset
    print("Evaluating the model on the test dataset...")
    test_loss, test_accuracy, report = model_utils.evaluate_model(model, test_data, test_x, test_y)

    # Visualize the training progress
    print("Visualizing the training progress...")
    model_utils.visualize_training(history)

# Map position numbers with letters
mapper = {i: file for i, file in enumerate(os.listdir(train_dir))}

# Make a single prediction
truth = "L"
input_image_path = test_dir + "/" + truth + "/3001.jpg"
print("Making a single prediction...")
model_utils.predict_image(model, input_image_path, truth, mapper)

# Evaluate the overall model performance
print("Evaluating the overall model performance...")
model_utils.evaluate_model_overall(model, data_test, testlabels0)


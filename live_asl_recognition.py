import os
import cv2
import numpy as np
import mediapipe as mp
from dataset_utils import DatasetUtils
from model_utils import ModelUtils

# Check if a saved model exists
saved_model_path = "models/ASL_model_20230617_205622.h5" #"models/ASL_model_20230617_185330.h5"
if os.path.exists(saved_model_path):
    print("Saved model found. Loading the model...")
    model = ModelUtils.load_saved_model(saved_model_path)
else:
    print("Saved model not found. Please make sure to train and save a model first.")
    exit()

# Set the path to the data directory.
train_dir = 'C:/Users/DhrCS/Documents/GitHub/American_Sign_Language/Data/Train_Original'

# Map position numbers with letters
mapper = {i: file for i, file in enumerate(os.listdir(train_dir))}

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands.Hands()
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation()

# Function to calculate hand bounding box with margin
def get_hand_bounding_box(hand_landmarks, width, height, margin=10):
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    # Add margin to the bounding box
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    # Ensure the coordinates are within the frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    return x_min, y_min, x_max, y_max

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Create a window for displaying hand_roi
cv2.namedWindow('Hand ROI', cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply selfie segmentation to extract the foreground (hand) and remove the background
    results_segmentation = mp_selfie_segmentation.process(frame_rgb)
    condition = np.stack((results_segmentation.segmentation_mask,) * 3, axis=-1) > 0.1
    foreground = np.where(condition, frame_rgb, 255).astype(np.uint8)

    # Convert the foreground frame to BGR
    foreground_bgr = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)

    # Detect hands using MediaPipe on the foreground frame
    results_hands = mp_hands.process(foreground_bgr)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Extract hand bounding box coordinates with margin
            x_min, y_min, x_max, y_max = get_hand_bounding_box(hand_landmarks, frame.shape[1], frame.shape[0], margin=50)

            # Extract hand region from the foreground frame
            hand_roi_img = foreground_bgr[y_min:y_max, x_min:x_max]
            hand_roi = cv2.resize(hand_roi_img, (40, 40))
            hand_roi = hand_roi.astype(np.float32) / 255.0
            hand_roi = np.expand_dims(hand_roi, axis=0)

            prediction = model.predict(hand_roi)
            predicted_class = np.argmax(prediction)
            predicted_move = mapper[predicted_class]

            # Draw hand bounding box and prediction on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, predicted_move, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
   
            # Convert the hand_roi frame to RGB for displaying
            hand_roi_display = cv2.cvtColor(hand_roi_img, cv2.COLOR_BGR2RGB)

            # Show hand_roi in the separate window
            cv2.imshow('Hand ROI', hand_roi_display)

    # Show the frame
    cv2.imshow('ASL Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

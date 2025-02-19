import cv2
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model

# Custom DepthwiseConv2D class to handle the unsupported 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Load the model with the custom DepthwiseConv2D layer
try:
    model = load_model('../models/keras_Model.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the labels
label_dict = {}
try:
    with open('../models/labels.txt', 'r', encoding='utf-8') as f:
        for line in f:
            label_id, label = line.strip().split(' ', 1)
            label_dict[int(label_id)] = label
except FileNotFoundError:
    print("Labels file not found.")
    exit()
except Exception as e:
    print(f"Error reading labels: {e}")
    exit()

# Preprocessing function to resize and normalize the input image
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to find and classify potential traffic signs
def detect_and_classify_signs(frame):
    # Convert the frame to HSV for color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for detecting traffic signs (red, for example)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask1 | mask2  # Combine both masks

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small contours that may not be traffic signs
        if cv2.contourArea(contour) < 500:
            continue

        # Calculate the aspect ratio and area to ensure the contour resembles a traffic sign
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if not (0.8 <= aspect_ratio <= 1.2):  # Traffic signs are generally square-shaped
            continue

        # Extract the region of interest and preprocess it for prediction
        roi = frame[y:y+h, x:x+w]
        preprocessed_roi = preprocess_image(roi)

        # Predict the class of the region
        predictions = model.predict(preprocessed_roi)
        class_id = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Filter predictions with a confidence threshold
        if confidence < 0.7:
            continue

        class_label = label_dict.get(class_id, "Unknown")

        # Ignore non-traffic sign classes (optional, depending on your model's classes)
        if class_label == "Unknown" or "Non-Sign" in class_label:
            continue

        # Draw the bounding box and label on the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# Real-time video capture from the webcam (device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam initialized. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect and classify traffic signs in the frame
    result_frame = detect_and_classify_signs(frame)

    # Show the frame with bounding boxes and labels
    cv2.imshow('Traffic Sign Recognition', result_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources after exiting the loop
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import statistics
import mediapipe as mp

# Load the trained model
model = load_model('models/model.keras')

# Get the number of output classes from the model's final layer
output_classes = model.output_shape[1]

# Generate the appropriate ASL label map based on how the model was trained
# For example: 25 classes might be A–Y (excluding Z), 24 excludes J and Z
if output_classes == 25:
    labels_map = [chr(i + 65) for i in range(26) if chr(i + 65) != 'Z']  # A–Y, exclude Z
elif output_classes == 24:
    labels_map = [chr(i + 65) for i in range(26) if chr(i + 65) not in ['J', 'Z']]
elif output_classes == 26:
    labels_map = [chr(i + 65) for i in range(26)]
else:
    raise ValueError(f"Unexpected number of output classes: {output_classes}")

# Initialize MediaPipe Hands solution for real-time hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

# Initialize a deque buffer to store recent predictions (for smoothing)
predictions = deque(maxlen=5)

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        break   # Exit if webcam frame is not available

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1) 

    # Convert frame to RGB (MediaPipe expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe to detect hand landmarks
    result = hands.process(frame_rgb)

    # This will hold the final smoothed prediction to display
    smoothed_letter = ""

    # If hands are detected in the frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            try:
                h, w, _ = frame.shape  # Get frame dimensions

                # Extract the bounding box coordinates around the hand
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                # Add padding to the bounding box and clip to image bounds
                x1, y1 = max(int(x_min) - 20, 0), max(int(y_min) - 20, 0)
                x2, y2 = min(int(x_max) + 20, w), min(int(y_max) + 20, h)

                # Extract region of interest (ROI) from the frame
                roi = frame[y1:y2, x1:x2]

                # Preprocess the ROI: grayscale → resize → normalize → reshape
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                normalized = resized / 255.0
                reshaped = normalized.reshape(1, 28, 28, 1)

                # Make a prediction using the CNN model
                prediction = model.predict(reshaped, verbose=0)
                predicted_index = np.argmax(prediction)

                # Convert prediction index to letter if valid
                if predicted_index < len(labels_map):
                    predicted_letter = labels_map[predicted_index]
                else:
                    predicted_letter = '?' # Safety fallback

                # Add prediction to buffer and calculate smoothed result
                predictions.append(predicted_letter)
                smoothed_letter = statistics.mode(predictions)

                # Draw the bounding box and prediction text on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Predicted: {smoothed_letter}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error in detection: {e}")  # Print error for debugging

    # Show the final frame with predictions
    cv2.imshow("ASL Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import statistics
import mediapipe as mp

# Load the trained model
model = load_model('models/model.keras')

# Determine output class count
output_classes = model.output_shape[1]

# Define label map based on class count
if output_classes == 25:
    labels_map = [chr(i + 65) for i in range(26) if chr(i + 65) != 'Z']  # A–Y, exclude Z
elif output_classes == 24:
    labels_map = [chr(i + 65) for i in range(26) if chr(i + 65) not in ['J', 'Z']]
elif output_classes == 26:
    labels_map = [chr(i + 65) for i in range(26)]
else:
    raise ValueError(f"Unexpected number of output classes: {output_classes}")

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

# Prediction smoothing buffer
predictions = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    smoothed_letter = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            try:
                h, w, _ = frame.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                # Pad the bounding box
                x1, y1 = max(int(x_min) - 20, 0), max(int(y_min) - 20, 0)
                x2, y2 = min(int(x_max) + 20, w), min(int(y_max) + 20, h)

                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                normalized = resized / 255.0
                reshaped = normalized.reshape(1, 28, 28, 1)

                # Predict
                prediction = model.predict(reshaped, verbose=0)
                predicted_index = np.argmax(prediction)

                if predicted_index < len(labels_map):
                    predicted_letter = labels_map[predicted_index]
                else:
                    predicted_letter = '?'

                predictions.append(predicted_letter)
                smoothed_letter = statistics.mode(predictions)

                # Draw output
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Predicted: {smoothed_letter}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error in detection: {e}")

    # Display webcam frame
    cv2.imshow("ASL Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
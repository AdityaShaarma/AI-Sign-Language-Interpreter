import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import statistics
import mediapipe as mp

# Load the trained model. Makes sure model is loaded once per session
@st.cache_resource
def load_trained_model():
    return load_model('models/model.keras')

model = load_trained_model()

# Get the number of output classes from the model's final layer
output_classes = model.output_shape[1]

# Generate the appropriate ASL label map based on how the model was trained
# For example: 25 classes might be A–Y (excluding Z), 24 excludes J and Z
if output_classes == 25:
    # A to Y (exclude Z)
    labels_map = list("ABCDEFGHIJKLMNOPQRSTUVWXY")
elif output_classes == 24:
    # A to Y (exclude J and Z)
    labels_map = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in ["J", "Z"]]
elif output_classes == 26:
    # A to Z
    labels_map = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
else:
    raise ValueError(f"Unexpected number of output classes: {output_classes}")

# Initialize MediaPipe Hands solution for real-time hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Streamlit UI
st.title("Live ASL Detection")
frame_placeholder = st.empty()

# Initialize session state variables to control streaming flow
if "stop_stream" not in st.session_state:
    st.session_state["stop_stream"] = False
if "start_stream" not in st.session_state:
    st.session_state["start_stream"] = False

# UI Buttons for Start and Stop to control video streaming
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Start"):
        st.session_state["start_stream"] = True
        st.session_state["stop_stream"] = False
with col2:
    if st.button("Stop"):
        st.session_state["stop_stream"] = True
        st.session_state["start_stream"] = False

# Proceed with real-time prediction only if Start is pressed and Stop is not
if st.session_state["start_stream"] and not st.session_state["stop_stream"]:
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    # Initialize a deque buffer to store recent predictions (for smoothing)
    predictions = deque(maxlen=5)

    # Main loop for real-time detection
    while not st.session_state["stop_stream"]:
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
                        predicted_letter = '?'  # Safety fallback

                    # Add prediction to buffer and calculate smoothed result
                    predictions.append(predicted_letter)
                    smoothed_letter = statistics.mode(predictions)

                    # Draw the bounding box and prediction text on the original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Predicted: {smoothed_letter}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error in detection: {e}")  # Print error for debugging

        # Display the processed video frame in the Streamlit app
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Release the webcam after stream stops
    cap.release()
    cv2.destroyAllWindows()
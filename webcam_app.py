import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/asl_model.keras')

# Map label indices to ASL letters (excluding J and Z which require motion)
labels_map = [chr(i + 65) for i in range(26)] # A-Z

# Start the webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

# Live detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define Region of Interest (ROI)
    x1, y1, x2, y2 = 100, 100, 800, 800
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI for prediction
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(reshaped)
    predicted_index = np.argmax(prediction)
    predicted_letter = labels_map[predicted_index]

    # Draw the ROI box and predicted letter on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f'Predicted: {predicted_letter}', (x1, y1 - 10),
                cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("ASL Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
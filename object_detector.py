<<<<<<< HEAD
import cv2
import numpy as np

# Load pre-trained model and configuration files
model = 'MobileNetSSD_deploy.caffemodel'
config = 'MobileNetSSD_deploy.prototxt'

# Load the object detection network
net = cv2.dnn.readNetFromCaffe(config, model)

# Define object classes the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize video capture (use 0 for webcam or provide a video filename)
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture frame!")
        break
    
    # Prepare frame for object detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Only consider confident detections
            idx = int(detections[0, 0, i, 1])  # Class index
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display frame
    cv2.imshow("Object Detector", frame)

    # Exit condition
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
capture.release()
cv2.destroyAllWindows()
=======
# Import Libraries
import cv2
import time
import mediapipe as mp
import numpy as np

# Initialize Pose and Hands models from Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load object detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Initialize video capture
capture = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    ret, frame = capture.read()

    frame = cv2.resize(frame, (800, 600))
    h, w = frame.shape[:2]

    # Object Detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract region of interest (ROI) for each person
            roi = frame[startY:endY, startX:endX]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Pose detection
            roi_rgb.flags.writeable = False
            pose_results = pose.process(roi_rgb)
            roi_rgb.flags.writeable = True

            # Hand detection
            hands_results = hands.process(roi_rgb)

            # Draw Pose landmarks
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    roi, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw Hand landmarks
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame[startY:endY, startX:endX] = roi

            # Draw bounding box for each detected person
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(frame, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Multi-Person Detection with Landmarks", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
>>>>>>> 58baf0093a8df7655d4a2f8557eec2fe57074b82

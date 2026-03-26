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
import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    """Calculate distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def detect_gesture(landmarks, image_shape):
    """Detect hand gesture based on landmark positions"""
    h, w, _ = image_shape
    
    # Convert landmarks to pixel coordinates
    points = []
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))
    
    # Define finger landmarks
    thumb_tip = points[4]
    thumb_mcp = points[2]
    index_tip = points[8]
    index_pip = points[6]
    index_mcp = points[5]
    middle_tip = points[12]
    middle_pip = points[10]
    middle_mcp = points[9]
    ring_tip = points[16]
    ring_pip = points[14]
    ring_mcp = points[13]
    pinky_tip = points[20]
    pinky_pip = points[18]
    pinky_mcp = points[17]
    wrist = points[0]

    # Calculate key distances and angles
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    thumb_angle = calculate_angle(thumb_tip, thumb_mcp, wrist)
    
    # Function to check if a finger is extended
    def is_finger_extended(tip, pip, mcp, threshold=50):
        return calculate_distance(tip, mcp) > threshold

    # Check extension state of each finger
    thumb_extended = thumb_angle > 30
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    
    # Gesture detection logic
    fingers_extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    
    # 1. Fist (no fingers extended)
    if not any(fingers_extended):
        return "Fist"
    
    # 2. Open Palm (all fingers extended)
    if all(fingers_extended):
        return "Open Palm"
    
    # 3. Peace Sign
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        if calculate_distance(index_tip, middle_tip) > 50:
            return "Peace"
    
    # 4. Pointing (only index extended)
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Pointing"
    
    # 5. Thumbs Up
    if thumb_extended and not any(fingers_extended[1:]):
        if thumb_tip[1] < wrist[1]:  # Check if thumb is above wrist
            return "Thumbs Up"
    
    # 6. Thumbs Down
    if thumb_extended and not any(fingers_extended[1:]):
        if thumb_tip[1] > wrist[1]:  # Check if thumb is below wrist
            return "Thumbs Down"
    
    # 7. OK Sign
    if thumb_index_dist < 30 and middle_extended and ring_extended and pinky_extended:
        return "OK"
    
    # 8. Rock On
    if index_extended and pinky_extended and not middle_extended and not ring_extended:
        return "Rock On"
    
    # 9. Spider-Man
    if index_extended and middle_extended and pinky_extended and not ring_extended:
        return "Spider-Man"
    
    # 10. Phone Call
    if thumb_extended and pinky_extended and not index_extended and not middle_extended and not ring_extended:
        return "Phone Call"
    
    # 11. Three Count
    if index_extended and middle_extended and ring_extended and not pinky_extended:
        return "Three"
    
    # 12. Four Count
    if index_extended and middle_extended and ring_extended and pinky_extended and not thumb_extended:
        return "Four"
        
    # 13. Gun Shape
    if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Gun Shape"
    
    # 14. Pinch
    if thumb_index_dist < 30 and not middle_extended and not ring_extended and not pinky_extended:
        return "Pinch"
    
    # 15. Live Long (Vulcan Salute)
    if index_extended and middle_extended and ring_extended and pinky_extended:
        if index_middle_dist > 30:
            return "Live Long"

    return "Unknown Gesture"

def draw_gesture_info(image, gesture):
    """Draw gesture information on the image"""
    # Add gesture name
    cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add gesture count
    cv2.putText(image, f"Recognized Gestures: 15", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for gesture smoothing
last_gesture = "Unknown Gesture"
gesture_counter = 0
GESTURE_SMOOTHING = 3  # Number of consecutive frames needed for a gesture change

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read camera")
        continue

    # Flip the image horizontally for selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw hand landmarks and detect gestures
    current_gesture = "Unknown Gesture"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Detect gesture
            current_gesture = detect_gesture(hand_landmarks, image.shape)
    
    # Gesture smoothing
    if current_gesture == last_gesture:
        gesture_counter += 1
    else:
        gesture_counter = 0
        
    if gesture_counter >= GESTURE_SMOOTHING:
        last_gesture = current_gesture

    # Draw gesture information
    draw_gesture_info(image, last_gesture)
    
    # Display the image
    cv2.imshow('Advanced Hand Gesture Detection', image)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
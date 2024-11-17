import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Threshold for thumb and index finger tip meeting
CLICK_THRESHOLD = 50.0  # You may need to adjust this depending on hand size and camera

# Initialize MediaPipe Hands model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Flip frame horizontally and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        hand_results = hands.process(rgb_frame)

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Track hand for mouse movement
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get index finger tip and thumb tip positions
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Map the finger positions to screen coordinates
                index_x = int(index_finger_tip.x * screen_width)
                index_y = int(index_finger_tip.y * screen_height)

                thumb_x = int(thumb_tip.x * screen_width)
                thumb_y = int(thumb_tip.y * screen_height)

                # Move the mouse pointer with the index finger tip
                pyautogui.moveTo(index_x, index_y)

                # Calculate the distance between the thumb tip and index finger tip
                distance = euclidean_distance((thumb_x, thumb_y), (index_x, index_y))

                # If the distance is below the threshold, simulate a click
                if distance < CLICK_THRESHOLD:
                    pyautogui.click()

                # Draw hand landmarks for visualization
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the webcam feed
        cv2.imshow("Hand Tracker", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

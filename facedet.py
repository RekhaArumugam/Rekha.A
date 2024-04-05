import cv2
import dlib
from scipy.spatial import distance as dist
import time

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define constants
EYE_AR_THRESH = 0.25  # Threshold for detecting eye closure
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames for eye closure
FRAME_CHECK_INTERVAL = 5  # Interval to check for drowsiness (in seconds)

# Initialize variables
counter = 0
drowsy = False

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract left and right eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate eye aspect ratio (EAR) for left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if EAR is below the threshold
        if ear < EYE_AR_THRESH:
            counter += 1

            if counter >= EYE_AR_CONSEC_FRAMES:
                drowsy = True
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            counter = 0
            drowsy = False

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Check for user input to exit the loop
    if key == ord("q"):
        break

    # Pause for FRAME_CHECK_INTERVAL seconds before checking for drowsiness again
    time.sleep(FRAME_CHECK_INTERVAL)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

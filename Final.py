import cv2
import mediapipe as mp
import math
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
model = keras.models.load_model('trained_model.h5')

# Load the scaler values
with open('scaler_values.pkl', 'rb') as file:
    mean_values, scale_values = pickle.load(file)

# Initialize the scaler with the loaded mean and scale values
scaler = StandardScaler()
scaler.mean_ = mean_values
scaler.scale_ = scale_values

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

try:
    # Open the video capture
    cap = cv2.VideoCapture(0)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        raise IOError("Failed to open video capture.")

    # Define the dimensions of the bounding box
    box_width = 170
    box_height = 400

    # Function to calculate the Euclidean distance between two landmarks
    def calculate_distance(landmark1, landmark2):
        return math.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2)

    # List to store the measurements
    measurements = []

    # Start the pose estimation process
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # Read a frame from the video capture
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)

            # Convert the frame from BGR to RGB for processing with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run pose estimation on the frame
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract specific landmarks for measurements
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                # Calculate distances between landmarks
                upper_body_distance = calculate_distance(left_shoulder, left_hip)
                leg_distance = calculate_distance(left_hip, left_ankle)
                hip_distance = calculate_distance(left_hip, right_hip)
                shoulder_distance = calculate_distance(left_shoulder, right_shoulder)

                # Append the measurements to the list
                measurements.append([upper_body_distance, leg_distance, hip_distance, shoulder_distance])

                # Convert the measurements list to a NumPy array
                measurements_arr = np.array(measurements)

                # Standardize the measurements
                standardized_measurements = scaler.transform(measurements_arr)

                # Find the measurement with the highest accuracy
                best_accuracy_measurement = max(standardized_measurements, key=lambda x: x[0])

                # Predict the pixel-to-cm ratio
                pixel_to_cm_ratio = model.predict(np.array([best_accuracy_measurement]))

                # Define the threshold value for measurement accuracy
                threshold = 0.5

                # Display the measurements on the frame
                cv2.putText(frame, f"Upper Body Distance: {upper_body_distance:.2f} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Leg Distance: {leg_distance:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(frame, f"Hip Distance: {hip_distance:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(frame, f"ShoulderDistance: {shoulder_distance:.2f} cm", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)

            # Display the frame with pose estimation
            cv2.imshow('Pose Estimation', frame)

            # Check for the 'q' key press to exit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    # Release the video capture and destroy all windows
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
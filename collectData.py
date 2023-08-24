import cv2
import mediapipe as mp
import math
import os
measurements = []
# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create a directory for data
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Define the dimensions of the bounding box
box_width = 170
box_height = 400

# Define the body parts
body_parts = ["hip", "shoulder", "upper_body", "leg"]

# Prompt the user to enter the number of persons
num_persons = int(input("Enter the number of persons: "))

# Create a file for all persons' measurements
measurements_file = os.path.join(data_folder, 'measurements.txt')
mediapipe_data_file = os.path.join(data_folder, 'mediapipe_data.txt')

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x)**2 + (landmark2.y - landmark1.y)**2)

# Open the measurements file in append mode
with open(measurements_file, 'a') as file, open(mediapipe_data_file, 'a') as mediapipe_file:
    # Loop through each person
    for person in range(1, num_persons + 1):
        # Prompt the user to enter real measurements for all body parts
        print(f"Enter real measurements for Person {person}:")
        measurements = {}
        for body_part in body_parts:
            measurement = input(f"{body_part}: ")
            measurements[body_part] = measurement

        # Write the measurements to the measurements file
        for body_part, measurement in measurements.items():
            file.write(f"{measurement}\n")

        # Open the video capture
        cap = cv2.VideoCapture(0)

        # Check if the video capture is successfully opened
        if not cap.isOpened():
            raise IOError("Failed to open video capture.")

        # Start the pose estimation process
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            measurements = []
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

                    # Draw a bounding box on the frame
                    frame_height, frame_width, _ = frame.shape
                    box_left = int((frame_width - box_width) / 2)
                    box_top = int((frame_height - box_height) / 2)
                    box_right = box_left + box_width
                    box_bottom = box_top + box_height
                    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

                    # Display the measurements on the frame
                    cv2.putText(frame, f"upper_body: {upper_body_distance:.2f} pixels", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"leg: {leg_distance:.2f} pixels", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"hip: {hip_distance:.2f} pixels", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"shoulder: {shoulder_distance:.2f} pixels", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Write the measurements to the mediapipe data file
                    measurements.append((hip_distance, shoulder_distance, upper_body_distance, leg_distance))
                # Draw the pose landmarks on the frame
                #mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display the frame
                cv2.imshow('MediaPipe Pose Estimation', frame)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture
        best_accuracy_measurement = max(measurements, key=lambda x: x[0])
        mediapipe_file.write(f"{best_accuracy_measurement[0]}\n")
        mediapipe_file.write(f"{best_accuracy_measurement[1]}\n")
        mediapipe_file.write(f"{best_accuracy_measurement[2]}\n")
        mediapipe_file.write(f"{best_accuracy_measurement[3]}\n")
        cap.release()
        cv2.destroyAllWindows()
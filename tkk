import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define the GPIO pins for the motor driver
MOTOR1_IN1 = 17  # Motor 1 forward
MOTOR1_IN2 = 27  # Motor 1 backward
MOTOR2_IN3 = 22  # Motor 2 forward
MOTOR2_IN4 = 23  # Motor 2 backward

# Setup the GPIO pins as output
GPIO.setup(MOTOR1_IN1, GPIO.OUT)
GPIO.setup(MOTOR1_IN2, GPIO.OUT)
GPIO.setup(MOTOR2_IN3, GPIO.OUT)
GPIO.setup(MOTOR2_IN4, GPIO.OUT)

def move_forward():
    GPIO.output(MOTOR1_IN1, GPIO.HIGH)
    GPIO.output(MOTOR1_IN2, GPIO.LOW)
    GPIO.output(MOTOR2_IN3, GPIO.HIGH)
    GPIO.output(MOTOR2_IN4, GPIO.LOW)

def turn_left():
    GPIO.output(MOTOR1_IN1, GPIO.LOW)
    GPIO.output(MOTOR1_IN2, GPIO.HIGH)
    GPIO.output(MOTOR2_IN3, GPIO.HIGH)
    GPIO.output(MOTOR2_IN4, GPIO.LOW)

def turn_right():
    GPIO.output(MOTOR1_IN1, GPIO.HIGH)
    GPIO.output(MOTOR1_IN2, GPIO.LOW)
    GPIO.output(MOTOR2_IN3, GPIO.LOW)
    GPIO.output(MOTOR2_IN4, GPIO.HIGH)

def stop_motors():
    GPIO.output(MOTOR1_IN1, GPIO.LOW)
    GPIO.output(MOTOR1_IN2, GPIO.LOW)
    GPIO.output(MOTOR2_IN3, GPIO.LOW)
    GPIO.output(MOTOR2_IN4, GPIO.LOW)

# Path to the template image (the object you want to detect)
template_image_path = r"C:\Users\pschi\Downloads\project logo.jpg"

# Load the template image in grayscale
template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

# Verify that the template image was loaded correctly
if template_image is None:
    print(f"Error: Template image not found at {template_image_path}")
    GPIO.cleanup()
    exit(1)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors in the template image using ORB
keypoints_template, descriptors_template = orb.detectAndCompute(template_image, None)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Verify that the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    GPIO.cleanup()
    exit(1)

# Create a BFMatcher object with Hamming distance and cross-check
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Known width of the object in centimeters (adjust based on your object)
KNOWN_WIDTH = 10.0  # cm

# Focal length of the camera (determine through calibration)
FOCAL_LENGTH = 500  # adjust this value based on your camera calibration

def calculate_distance(perceived_width):
    """Calculate the distance from the camera to the object."""
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in the frame using ORB
    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

    if descriptors_frame is not None and len(descriptors_frame) > 0:
        # Match descriptors between the template image and the frame
        matches = bf.knnMatch(descriptors_template, descriptors_frame, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Minimum number of good matches required to find the object
        MIN_MATCH_COUNT = 10

        if len(good_matches) >= MIN_MATCH_COUNT:
            # Extract location of good matches
            src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Find homography matrix
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Get the corners of the template image
            h, w = template_image.shape
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            if H is not None:
                # Transform the corners to find the location in the frame
                transformed_corners = cv2.perspectiveTransform(corners, H)
                frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # Calculate the perceived width of the object in the frame
                perceived_width = np.linalg.norm(transformed_corners[0] - transformed_corners[3])

                # Calculate the distance to the object
                distance = calculate_distance(perceived_width)

                if distance <= 60:  # Check if the object is within 60 cm
                    print(f"Object Detected at distance: {distance:.2f} cm")

                    # Calculate the center of the detected object
                    object_center = np.mean(transformed_corners, axis=0).flatten()

                    # Divide the frame into three portions
                    frame_width = frame.shape[1]
                    left_boundary = frame_width / 3
                    right_boundary = 2 * frame_width / 3

                    # Check which portion the object is in
                    if object_center[0] < left_boundary:
                        print("Turning Left")
                        turn_left()
                    elif object_center[0] > right_boundary:
                        print("Turning Right")
                        turn_right()
                    else:
                        print("Moving Forward")
                        move_forward()
                else:
                    print(f"Object detected but out of range: {distance:.2f} cm")
                    stop_motors()
            else:
                print("Homography not found. Object not detected.")
                stop_motors()
        else:
            print(f"Not enough good matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")
            stop_motors()

    # Draw matches for visualization
    match_img = cv2.drawMatches(template_image, keypoints_template, frame, keypoints_frame, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the frame with detected object
    cv2.imshow('Frame', match_img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()

import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO('C:/Users/YOURDESKTOP/Desktop/DogIdwithYOLO/.venv/runs/train/dog_detector4/weights/best.pt')

video_path = 'C:/Users/YOURDESKTOP/Desktop/DogIdwithYOLO/.venv/videos/test_video.mp4'
cap = cv2.VideoCapture(video_path)

# Increase resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

prev_center = None
alert_state = "normal"  # normal, yellow, red
threat_level = "none"  # none, attack, fast

# Speed thresholds
speed_threshold_normal = 20  # Speed limit for normal movement
speed_threshold_fast = 50   # Speed limit for fast movement

# Function to detect likely human silhouettes
def is_likely_human_silhouette(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / float(w)
    return aspect_ratio > 1.6  # High ratio for bipedal humans

# Function to display alert messages
def show_alert(message, color=(0, 255, 255)):
    """Displays an alert message on a small screen."""
    alert_img = np.zeros((100, 500, 3), dtype=np.uint8)  # Create the alert screen
    cv2.putText(alert_img, message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # Resize the font
    cv2.imshow("Alert", alert_img)  # Show the alert message on the screen

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # Detect dog in the video frame
    dog_detected = False
    current_threat_level = "none"  # Temporary threat level

    # Find contours for motion and silhouette detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if is_likely_human_silhouette(contour):  # If a human silhouette is detected
            current_threat_level = "human"
            alert_state = "yellow"
            break  # If a human is detected, don't proceed with dog detection

    for box in results.boxes:
        cls = int(box.cls[0])  # Get the class (0 is for dog class)
        if cls == 0:  # If dog is detected
            dog_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center coordinates

            # Movement control
            if prev_center:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5

                # Speed analysis: Yellow alert for fast movement
                if distance > speed_threshold_fast:
                    current_threat_level = "fast"
                    alert_state = "yellow"
                elif distance > speed_threshold_normal:
                    current_threat_level = "normal"

                # Red alert: aggressive approach
                if (x2 - x1) > 200 and current_threat_level != "attack":  # Large and fast approaching dog
                    current_threat_level = "attack"
                    alert_state = "red"

            prev_center = (cx, cy)

            # Draw rectangle on the video
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the box
            cv2.putText(frame, "Dog", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display speed and threat level on the frame
            if current_threat_level == "attack":
                cv2.putText(frame, "Threat: Attack", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif current_threat_level == "fast":
                cv2.putText(frame, "Threat: FAST", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            break

    # Alert screen
    if alert_state == "yellow":
        show_alert("YELLOW ALERT: Running!", (0, 255, 255))  # Show yellow alert
    elif alert_state == "red":
        show_alert("RED ALERT: ATTACK!", (0, 0, 255))  # Show red alert
    elif alert_state == "normal" and dog_detected:
        show_alert("Status Normal", (0, 255, 0))  # Show green alert for normal status

    # Normalize the status
    if alert_state == "yellow" and not dog_detected:
        alert_state = "normal"
        prev_center = None

    # Show video frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Exit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()  # Close all windows

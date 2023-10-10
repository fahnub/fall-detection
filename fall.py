import mediapipe as mp
import numpy as np
import cv2


def calculate_angle(a,b,c):
    a = np.array(a) #First
    b = np.array(b) #Mid
    c = np.array(c) #End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

fcc = cv2.VideoWriter_fourcc(*'XVID')
size = (1920, 1080)
video_number = 5
path = f"media/fall-detection/{video_number}"
video_output = cv2.VideoWriter(f"{path}.avi", fcc, 60, size)
cap = cv2.VideoCapture(f"{path}.mp4")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        result = pose.process(frame)

        if result.pose_landmarks is not None:
            landmarks = result.pose_landmarks.landmark

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)

            if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.3 or landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.3:
                if angle > 175:
                    position = "Normal"
                else:
                    position = "Fall Detected"
                    message = "EMERGENCY!!!"
                    cv2.putText(frame, message, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.putText(frame, f"Position: {position}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

            video_output.write(frame)

        cv2.imshow('Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    video_output.release()
    cap.release()
    cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def extract(video_path, out_dir="poses"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        canvas = frame * 0
        if result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                canvas,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imwrite(f"{out_dir}/{idx:04d}.png", canvas)
        idx += 1

    cap.release()
    print("✅ Pose extracted")

if __name__ == "__main__":
    extract("motion.mp4")

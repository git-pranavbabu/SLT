import cv2
import os

video_path = 'raw_videos/Nc7rSopCpI8.mp4'
if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

print("Attempting to read first 5 frames sequentially:")
for i in range(5):
    ret, frame = cap.read()
    print(f"Frame {i}: ret={ret}, shape={frame.shape if ret else None}")

print("\nAttempting seek to frame 30:")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
print(f"Frame 30 after seek: ret={ret}")

cap.release()

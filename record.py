import cv2 as cv
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# --- Settings ---
OUT_DIR = Path("captures_ME")   # where to save videos
CAM_INDEX = 1                   # change if you have multiple cameras
MIN_TOGGLE_SEC = 0.30           # debounce for record toggle

OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

# Try to get FPS & frame size from camera
fps = cap.get(cv.CAP_PROP_FPS)
if fps is None or fps <= 1 or np.isnan(fps):
    fps = 30.0  # fallback if the camera doesn't report FPS

width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)

# Choose a codec/extension pair that works on你的系統
# Windows/一般相容: 'mp4v' + .mp4 ；若不行可改 'XVID' + .avi
fourcc = cv.VideoWriter_fourcc(*'mp4v')

writer = None
recording = False
last_toggle = 0.0

print("Press 'r' to START/STOP recording, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # If recording, write current frame
    if recording and writer is not None:
        writer.write(frame)
        # Draw REC indicator
        cv.circle(frame, (20, 20), 8, (0,0,255), -1)
        cv.putText(frame, "REC", (35, 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv.LINE_AA)

    # Show live view
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 如果你偏好灰階預覽可用 gray；要彩色就把下行改成 frame
    cv.imshow('frame', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('r'):
        now = time.time()
        if now - last_toggle >= MIN_TOGGLE_SEC:
            if not recording:
                # Start recording
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                video_path = OUT_DIR / f"vid_{ts}.mp4"
                writer = cv.VideoWriter(str(video_path), fourcc, fps, frame_size)
                if not writer.isOpened():
                    print("Failed to open VideoWriter. Try a different codec/extension.")
                else:
                    print(f"Recording started -> {video_path.name} @ {fps} FPS, {frame_size}")
                    recording = True
            else:
                # Stop recording
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                    print("Recording stopped.")
            last_toggle = now

# Cleanup
if writer is not None:
    writer.release()
cap.release()
cv.destroyAllWindows()
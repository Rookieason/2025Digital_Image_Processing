import cv2
import numpy as np
from utils.util import process1, process2, process3, process4
from utils.process_for_reconstruction import image_process_original, image_process_temporal_smoothing

def main():
    prev_gains = None
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p1 = image_process_original(frame)
        p2, prev_gains = image_process_temporal_smoothing(frame, prev_gains=prev_gains)
        cv2.imshow("original", p1)
        cv2.imshow("after", p2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

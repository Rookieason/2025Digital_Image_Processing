import cv2
import numpy as np
from utils.util import process1, process2, process3


def main():
    cap = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()

        p1 = process1(frame)
        p2 = process2(frame)
        p3 = process3(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("P1", p1)
        cv2.imshow("P2", p2)
        cv2.imshow("P3", p3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

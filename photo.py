import cv2

def main():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("asdfasdf", frame)

        key = cv2.waitKey(1) & 0xFF

        # 按 C 拍照
        if key == ord("c") or key == ord("C"):
            cv2.imwrite("white.jpg", frame)
            print("Saved white.jpg")

        # 按 Q 離開
        if key == ord("q") or key == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

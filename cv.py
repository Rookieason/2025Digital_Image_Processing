import cv2
import numpy as np
from utils.usb_baseline_pipeline import USBBaselinePipeline
from utils.build import build

def load_calibration():
    data = np.load("utils/calib.npz")
    camera_matrix = data.get("camera_matrix", None) 
    dist_coeffs = data.get("dist_coeffs", None)
    vignette_mask = data.get("vignette_mask", None)
    crf_lut = data.get("crf_lut", None)

    return camera_matrix, dist_coeffs, vignette_mask, crf_lut


def main():
    build()
    cap = cv2.VideoCapture(1)
    camera_matrix, dist_coeffs, vignette_mask, crf_lut = load_calibration()
    pipeline = USBBaselinePipeline(
        gamma=2.2,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        vignette_mask=vignette_mask,
        crf_lut=crf_lut,
        exposure_smooth_alpha=0.1
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        baseline = pipeline.process(frame)
        cv2.imshow("Baseline Processed(Gray)", baseline)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
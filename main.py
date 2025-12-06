import cv2
import numpy as np
import time
from utils.usb_baseline_pipeline import USBBaselinePipeline
from utils.build import build
from utils.usb_improvedv1_pipeline import USBImprovedv1Pipeline
from utils.usb_baseline_pipeline_fast import USBBaselinePipelineFast
from utils.Exposure_Compensation_v5 import integrate_into_pipeline
from utils.giveup import preprocess

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
    # pipeline = USBBaselinePipelineFast(
    #     gamma=2.2,
    #     camera_matrix=camera_matrix,
    #     dist_coeffs=dist_coeffs,
    #     vignette_mask=vignette_mask,
    #     crf_lut=crf_lut,
    #     exposure_smooth_alpha=0.1,
    #     denoise_method="gaussian"
    # )
    # pipelinev1 = USBImprovedv1Pipeline(
    #     gamma=2.2,
    #     camera_matrix=camera_matrix,
    #     dist_coeffs=dist_coeffs,
    #     vignette_mask=vignette_mask,
    #     crf_lut=crf_lut,
    #     exposure_smooth_alpha=0.1,
    #     denoise_method="gaussian"
    # )
    # --- FPS 計算 ---
    prev_time = time.time()
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # pipeline = integrate_into_pipeline(pipeline)
        # pipeline
        # improved = pipeline.process(frame)

        improved = preprocess(
            frame,
            camera_matrix,
            dist_coeffs,
            vignette_mask,
            crf_lut
        )
        # --- 計算 FPS (每秒更新一次) ---
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed >= 1.0:  # 每秒更新
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = current_time

        # --- 顯示 FPS ---
        display = improved.copy()

        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 30),                # 左上角
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)       # 綠色字

        cv2.imshow("Improved", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

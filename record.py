import cv2
import numpy as np
from utils.build import build
from utils.usb_improvedv1_pipeline import USBImprovedv1Pipeline
from utils.usb_baseline_pipeline_fast import USBBaselinePipelineFast

from utils.Exposure_Compensation_v1 import integrate_into_pipeline

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

    # --- 取得輸入影像大小 & FPS ---
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    # --- 三個影片輸出（mp4／h264） ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_raw = cv2.VideoWriter("raw.mp4", fourcc, fps, (width, height))
    # out_baseline = cv2.VideoWriter("baseline.mp4", fourcc, fps, (width, height))
    # out_improved = cv2.VideoWriter("improved.mp4", fourcc, fps, (width, height))

    # --- 初始化模型 ---
    camera_matrix, dist_coeffs, vignette_mask, crf_lut = load_calibration()

    pipeline = USBBaselinePipelineFast(
        gamma=2.2,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        vignette_mask=vignette_mask,
        crf_lut=crf_lut,
        exposure_smooth_alpha=0.1,
        denoise_method="gaussian"
    )
    # pipeline = USBImprovedv1Pipeline(
    #     gamma=2.2,
    #     camera_matrix=camera_matrix,
    #     dist_coeffs=dist_coeffs,
    #     vignette_mask=vignette_mask,
    #     crf_lut=crf_lut,
    #     exposure_smooth_alpha=0.1,
    #     denoise_method="gaussian"
    # )
    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cnt = cnt + 1
        # --- 1. 原始 frame ---
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out_raw.write(frame)

        # # --- 2. baseline (未穩定) ---
        # baseline = pipeline.process(frame)
        # # pipeline 出來通常是 gray，需要轉回 3 channel 才能寫入 mp4
        # if len(baseline.shape) == 2:
        #     baseline_color = cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR)
        # else:
        #     baseline_color = baseline
        # out_baseline.write(baseline_color)


        # --- 3. improved ---
        # pipeline = integrate_into_pipeline(pipeline)
        # improved = pipeline.process(frame)
        # pipeline 出來通常是 gray，需要轉回 3 channel 才能寫入 mp4
        # if len(improved.shape) == 2:
        #     improved_color = cv2.cvtColor(improved, cv2.COLOR_GRAY2BGR)
        # else:
        #     improve_color = improved
        # out_improved.write(improved_color)

        cv2.imshow("Original", frame)
        # --- 按 q 離開 ---
        if cv2.waitKey(1) & 0xFF == ord("q") or cnt > 2000:
            break

    cap.release()
    out_raw.release()
    # out_baseline.release()
    # out_improved.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np


# from utils.usb_baseline_pipeline_fast import USBBaselinePipelineFast
from utils.usb_pipeline import USBBaselinePipelineFast
from utils.Exposure_Compensation_v1 import integrate_into_pipeline

def load_calibration():
    data = np.load("utils/calib.npz")
    camera_matrix = data.get("camera_matrix", None) 
    dist_coeffs = data.get("dist_coeffs", None)
    vignette_mask = data.get("vignette_mask", None)
    crf_lut = data.get("crf_lut", None)

    return camera_matrix, dist_coeffs, vignette_mask, crf_lut

def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
    print(w, h)

    camera_matrix, dist_coeffs, vignette_mask, crf_lut = load_calibration()
    pipeline = USBBaselinePipelineFast(
        gamma=2.2,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        vignette_mask=vignette_mask,
        crf_lut=crf_lut,
        exposure_smooth_alpha=0.1,
        denoise_method="gaussian",
    )

    pipeline = integrate_into_pipeline(pipeline)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = pipeline.process(frame)
        if len(gray.shape) == 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        writer.write(gray)
    cap.release()
    writer.release()

if __name__ == "__main__":
    input_video = "i.mp4"
    output_video = "i_online.mp4"
    process_video(input_video, output_video)

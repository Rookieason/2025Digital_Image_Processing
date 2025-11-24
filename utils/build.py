import cv2
import numpy as np

def build_srgb_inverse_lut():
    """
    產生 0~255 -> 線性亮度 的 sRGB inverse LUT
    回傳 shape = (256,) 的 float32 array, 範圍約 0~1
    """
    lut = np.arange(256, dtype=np.float32) / 255.0
    # sRGB -> linear
    mask = lut <= 0.04045
    lut[mask] = lut[mask] / 12.92
    lut[~mask] = ((lut[~mask] + 0.055) / 1.055) ** 2.4
    return lut
def bt709_luma(img):
    """
    img: float32 BGR (0~1)
    return: luma (0~1)
    """
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build_vignetting_mask(
    white_img_path,
    camera_matrix,
    dist_coeffs
):
    """
    讀取 white.jpg → undistort → luma → normalize → vignette_mask
    """
    # ---- 1. 讀白紙 ----
    img = cv2.imread(white_img_path)
    if img is None:
        raise FileNotFoundError("Cannot load white image:", white_img_path)

    img = img.astype(np.float32) / 255.0

    # ---- 2. Undistort ----
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # ---- 3. Luma (BT.709) ----
    Y = bt709_luma(undistorted)

    # ---- 4. Gaussian 平滑避免局部噪聲 ----
    Y_blur = cv2.GaussianBlur(Y, (31, 31), 0)

    # ---- 5. Normalize：中心亮度 = 1 ----
    h, w = Y_blur.shape
    cx, cy = w // 2, h // 2
    center_val = float(Y_blur[cy, cx])

    if center_val < 1e-6:
        raise ValueError("Center luminance too small for vignetting normalization.")

    vignette = Y_blur / center_val  # center = 1.0

    # ---- 6. 限制範圍避免太極端 ----
    vignette = np.clip(vignette, 0.3, 1.0)

    return vignette.astype(np.float32)

def build():
    # --- 1. Camera intrinsic matrix K ---
    camera_matrix = np.array([
        [652.30864763, 0.0,         330.21432284],
        [0.0,          652.56596763, 251.691599  ],
        [0.0,          0.0,          1.0         ]
    ], dtype=np.float32)

    # --- 2. Distortion coefficients (k1..k6, p1, p2) ---
    dist_coeffs = np.array([
        -11.391366,  # k1
        41.871954,   # k2
        -0.003851,   # p1 (tangential)
        0.001381,    # p2 (tangential)
        -21.659587,  # k3
        -11.206537,  # k4
        39.859513,   # k5
        -14.928878   # k6
    ], dtype=np.float32)

    vignette_mask = build_vignetting_mask(
        "white.jpg",
        camera_matrix,
        dist_coeffs
    )

    # --- 3. sRGB inverse CRF LUT ---
    crf_lut = build_srgb_inverse_lut()  # shape (256,)

    # --- 4. 存成 calib.npz（不存 vignette_mask）---
    np.savez(
        "calib.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        crf_lut=crf_lut,
        vignette_mask=vignette_mask
    )
    print("Saved calib.npz with camera_matrix, dist_coeffs, vignette_mask, crf_lut.")
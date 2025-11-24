import numpy as np
import cv2

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, x*12.92, (1 + a)*np.power(np.maximum(x, 0.0), 1/2.4) - a)

def white_balance_sog(frame_bgr: np.ndarray, p: int = 6, thresh: float = 0.0) -> np.ndarray:
    eps = 1e-6
    bgr = frame_bgr.astype(np.float32) / 255.0
    bgr_lin = _srgb_to_linear(bgr)
    B, G, R = bgr_lin[..., 0], bgr_lin[..., 1], bgr_lin[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    if thresh > 0.0:
        mask = L > thresh
    else:
        mask = slice(None)

    def minkowski_mean(x):
        if isinstance(mask, slice):
            return (np.mean(np.power(np.maximum(x, eps), p)) + eps) ** (1.0 / p)
        else:
            xm = x[mask]
            if xm.size == 0:
                return (np.mean(np.power(np.maximum(x, eps), p)) + eps) ** (1.0 / p)
            return (np.mean(np.power(np.maximum(xm, eps), p)) + eps) ** (1.0 / p)

    Er = minkowski_mean(R)
    Eg = minkowski_mean(G)
    Eb = minkowski_mean(B)
    gains = np.array([1.0 / Eb, 1.0 / Eg, 1.0 / Er], dtype=np.float32)
    gains /= np.mean(gains) + eps
    balanced_lin = bgr_lin * gains[None, None, :]
    return balanced_lin.astype(np.float32)

def white_balance_sog_temporal(
    frame_bgr: np.ndarray,
    p: int = 6,
    thresh: float = 0.0,
    prev_gains: np.ndarray | None = None,
    alpha: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-6
    bgr = frame_bgr.astype(np.float32) / 255.0
    bgr_lin = _srgb_to_linear(bgr)
    B, G, R = bgr_lin[..., 0], bgr_lin[..., 1], bgr_lin[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    if thresh > 0.0:
        mask = L > thresh
    else:
        mask = slice(None)

    def minkowski_mean(x):
        if isinstance(mask, slice):
            return (np.mean(np.power(np.maximum(x, eps), p)) + eps) ** (1.0 / p)
        else:
            xm = x[mask]
            if xm.size == 0:
                return (np.mean(np.power(np.maximum(x, eps), p)) + eps) ** (1.0 / p)
            return (np.mean(np.power(np.maximum(xm, eps), p)) + eps) ** (1.0 / p)

    Er = minkowski_mean(R)
    Eg = minkowski_mean(G)
    Eb = minkowski_mean(B)
    gains_inst = np.array([1.0 / Eb, 1.0 / Eg, 1.0 / Er], dtype=np.float32)
    gains_inst /= np.mean(gains_inst) + eps

    if prev_gains is not None:
        prev_gains = np.asarray(prev_gains, dtype=np.float32).reshape(3)
        gains = alpha * prev_gains + (1.0 - alpha) * gains_inst
    else:
        gains = gains_inst

    balanced_lin = bgr_lin * gains[None, None, :]
    return balanced_lin.astype(np.float32), gains

def denoise_luma_linear(bgr_linear: np.ndarray, strength: float = 0.4) -> np.ndarray:
    M = np.array([[0.2126, 0.7152, 0.0722],
                  [-0.1146, -0.3854, 0.5],
                  [0.5, -0.4542, -0.0458]], dtype=np.float32)
    rgb_linear = bgr_linear[..., ::-1]
    ycbcr = rgb_linear @ M.T
    Y = ycbcr[..., 0].astype(np.float32)
    Yf = cv2.bilateralFilter(Y, d=3, sigmaColor=0.05*(1+strength), sigmaSpace=1+int(1*strength))
    ycbcr[..., 0] = Yf
    rgb_out = ycbcr @ np.linalg.inv(M.T)
    bgr_out = rgb_out[..., ::-1]
    return bgr_out

def color_correction_ccm(bgr_linear: np.ndarray, ccm: np.ndarray | None = None) -> np.ndarray:
    if ccm is None:
        ccm = np.array([[1.05, -0.05,  0.00],
                        [-0.02, 1.02,  0.00],
                        [0.00, -0.03,  1.03]], dtype=np.float32)
    H, W, _ = bgr_linear.shape
    x = bgr_linear.reshape(-1, 3) @ ccm.T
    return x.reshape(H, W, 3)

def enhance_edges_luma_linear(
    bgr_linear: np.ndarray,
    amount: float = 0.3,
    radius: int = 2,
    thresh: float = 0.02,
) -> np.ndarray:
    M = np.array([[0.2126, 0.7152, 0.0722],
                  [-0.1146, -0.3854, 0.5],
                  [0.5, -0.4542, -0.0458]], dtype=np.float32)
    MinvT = np.linalg.inv(M).T
    ycbcr = bgr_linear @ M.T
    Y = ycbcr[..., 0].astype(np.float32)
    ksize = 2 * radius + 1
    Y_blur = cv2.GaussianBlur(Y, (ksize, ksize), 0)
    high = Y - Y_blur
    mask = np.abs(high) > thresh
    Y_sharp = Y.copy()
    Y_sharp[mask] = Y[mask] + amount * high[mask]
    Y_sharp = np.clip(Y_sharp, 0.0, 1.0)
    ycbcr[..., 0] = Y_sharp
    bgr_sharp = ycbcr @ MinvT
    return bgr_sharp

def image_process_original(frame: np.ndarray) -> np.ndarray:
    wb = white_balance_sog(frame)
    dll = denoise_luma_linear(wb)
    ccm = color_correction_ccm(dll)
    return ccm

def image_process_temporal_smoothing(frame: np.ndarray,
    prev_gains: np.ndarray | None = None,
    alpha: float = 0.9,
    p: int = 6,
    thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    wb_lin, gains = white_balance_sog_temporal(
        frame,
        p=p,
        thresh=thresh,
        prev_gains=prev_gains,
        alpha=alpha,
    )
    dll = denoise_luma_linear(wb_lin)
    ccm = color_correction_ccm(dll)
    return ccm, gains

def tt(frame: np.ndarray,
    prev_gains: np.ndarray | None = None,
    alpha: float = 0.9,
    p: int = 6,
    thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    wb_lin, gains = white_balance_sog_temporal(
        frame,
        p=p,
        thresh=thresh,
        prev_gains=prev_gains,
        alpha=alpha,
    )
    dll = denoise_luma_linear(wb_lin)
    sharp = enhance_edges_luma_linear(dll)
    # ccm = color_correction_ccm(sharp)
    return sharp, gains

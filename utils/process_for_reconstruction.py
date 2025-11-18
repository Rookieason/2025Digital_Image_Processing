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

def tonemap_reinhard_luma_preserve(bgr_linear: np.ndarray, key: float = 0.18, white: float | None = None) -> np.ndarray:

    eps = 1e-6
    B, G, R = bgr_linear[..., 0], bgr_linear[..., 1], bgr_linear[..., 2]

    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    Lm_bar = np.exp(np.mean(np.log(eps + L)))

    Lm = (key / (Lm_bar + eps)) * L

    if white is not None:
        Ld = (Lm * (1.0 + Lm / (white * white))) / (1.0 + Lm)
    else:
        Ld = Lm / (1.0 + Lm)
    scale = Ld / (L + eps)
    out_lin = bgr_linear * scale[..., None]

    out_srgb = _linear_to_srgb(out_lin)
    out_srgb = np.clip(out_srgb, 0.0, 1.0)
    return (out_srgb * 255.0 + 0.5).astype(np.uint8)

def white_balance_sog_temporal(
    frame_bgr: np.ndarray,
    p: int = 6,
    thresh: float = 0.0,
    prev_gains: np.ndarray | None = None,
    alpha: float = 0.9,
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

    # per-frame instantaneous gains
    gains_inst = np.array([1.0 / Eb, 1.0 / Eg, 1.0 / Er], dtype=np.float32)
    gains_inst /= np.mean(gains_inst) + eps

    # temporal smoothing on gains (EMA)
    if prev_gains is not None:
        prev_gains = np.asarray(prev_gains, dtype=np.float32).reshape(3)
        gains = alpha * prev_gains + (1.0 - alpha) * gains_inst
    else:
        gains = gains_inst

    # 可選：避免偶發 frame 拉太炸
    # gains = np.clip(gains, 0.5, 2.0)

    balanced_lin = bgr_lin * gains[None, None, :]
    return balanced_lin.astype(np.float32), gains


def denoise_luma_linear(bgr_linear: np.ndarray, strength: float = 0.4) -> np.ndarray:
    # 轉 YCbCr(Rec.709) → 僅平滑 Y
    M = np.array([[0.2126, 0.7152, 0.0722],
                  [-0.1146, -0.3854, 0.5],
                  [0.5, -0.4542, -0.0458]], dtype=np.float32)
    ycbcr = bgr_linear @ M.T
    Y = ycbcr[..., 0].astype(np.float32)
    Yf = cv2.bilateralFilter(Y, d=3, sigmaColor=0.05*(1+strength), sigmaSpace=1+int(1*strength))
    ycbcr[..., 0] = Yf
    return ycbcr @ np.linalg.inv(M.T)

def color_correction_ccm(bgr_linear: np.ndarray, ccm: np.ndarray | None = None) -> np.ndarray:
    if ccm is None:
        # 近似 sRGB 的輕量校正（BGR順序），你可換成標定矩陣
        ccm = np.array([[1.05, -0.05,  0.00],
                        [-0.02, 1.02,  0.00],
                        [0.00, -0.03,  1.03]], dtype=np.float32)
    H, W, _ = bgr_linear.shape
    x = bgr_linear.reshape(-1, 3) @ ccm.T
    return x.reshape(H, W, 3)

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
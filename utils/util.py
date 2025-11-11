import numpy as np
import cv2
from vidgear.gears.stabilizer import Stabilizer
def white_balance_ffcc(frame_bgr: np.ndarray) -> np.ndarray:

    eps = 1e-6

    bgr = frame_bgr.astype(np.float32) / 255.0
    a = 0.055
    srgb_to_linear = lambda x: np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)
    bgr_lin = srgb_to_linear(bgr)

    B, G, R = bgr_lin[..., 0], bgr_lin[..., 1], bgr_lin[..., 2]
    Gs = np.maximum(G, eps)
    Rs = np.maximum(R, eps)
    Bs = np.maximum(B, eps)

    rg = np.log(np.maximum(Rs / Gs, eps))
    bg = np.log(np.maximum(Bs / Gs, eps))

    bins = 64
    rg_min, rg_max = -3.0, 3.0
    bg_min, bg_max = -3.0, 3.0

    rg_idx = np.clip(((rg - rg_min) / (rg_max - rg_min) * bins).astype(np.int32), 0, bins - 1)
    bg_idx = np.clip(((bg - bg_min) / (bg_max - bg_min) * bins).astype(np.int32), 0, bins - 1)

    flat = (bg_idx.ravel() * bins + rg_idx.ravel())
    hist = np.bincount(flat, minlength=bins * bins).astype(np.float32).reshape(bins, bins)
    hist = cv2.blur(hist, (3, 3))

    bg_p, rg_p = np.unravel_index(np.argmax(hist), hist.shape)
    rg_peak = rg_min + (rg_p + 0.5) * (rg_max - rg_min) / bins
    bg_peak = bg_min + (bg_p + 0.5) * (bg_max - bg_min) / bins

    c_R = np.exp(rg_peak)
    c_G = 1.0
    c_B = np.exp(bg_peak)

    gains = np.array([1.0 / c_B, 1.0 / c_G, 1.0 / c_R], dtype=np.float32)
    gains /= (np.mean(gains) + eps)  

    balanced_lin = bgr_lin * gains[None, None, :]
    return balanced_lin.astype(np.float32)


def tonemap_reinhard(bgr_linear: np.ndarray) -> np.ndarray:

    mapped = bgr_linear / (1.0 + np.maximum(bgr_linear, 0.0))


    a = 0.055
    linear_to_srgb = lambda x: np.where(x <= 0.0031308, x*12.92, (1 + a)*np.power(np.maximum(x, 0.0), 1/2.4) - a)
    out_srgb = linear_to_srgb(mapped)

    out_srgb = np.clip(out_srgb, 0.0, 1.0)
    out_u8 = (out_srgb * 255.0 + 0.5).astype(np.uint8)
    return out_u8


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

def denoise_chroma_linear(bgr_linear: np.ndarray, ksize: int = 3) -> np.ndarray:
    M = np.array([[0.2126, 0.7152, 0.0722],
                  [-0.1146, -0.3854, 0.5],
                  [0.5, -0.4542, -0.0458]], dtype=np.float32)
    ycbcr = bgr_linear @ M.T
    ycbcr[..., 1] = cv2.medianBlur(ycbcr[..., 1].astype(np.float32), ksize)
    ycbcr[..., 2] = cv2.medianBlur(ycbcr[..., 2].astype(np.float32), ksize)
    return ycbcr @ np.linalg.inv(M.T)


def process1(frame: np.ndarray) -> np.ndarray:

    dll = denoise_luma_linear(frame)
    wb = white_balance_sog(dll)
    ccm = color_correction_ccm(wb)
    tm = tonemap_reinhard_luma_preserve(ccm)

    return tm

def process2(frame: np.ndarray) -> np.ndarray:

    wb = white_balance_sog(frame)
    tm = tonemap_reinhard_luma_preserve(wb)

    return tm

def process3(frame: np.ndarray) -> np.ndarray:

    wb = white_balance_ffcc(frame)
    tm = tonemap_reinhard(wb)

    return tm

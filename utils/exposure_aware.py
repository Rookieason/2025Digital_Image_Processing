import numpy as np
import cv2
from dataclasses import dataclass

# ----------------------------------------
# 基礎色彩轉換
# ----------------------------------------

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, x*12.92, (1 + a)*np.power(np.maximum(x, 0.0), 1/2.4) - a)

# ----------------------------------------
# White Balance：你原有的版本（有 temporal）
# ----------------------------------------

def white_balance_sog(
    frame_bgr: np.ndarray,
    p: int = 6,
    thresh: float = 0.0,
) -> np.ndarray:
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

# ----------------------------------------
# Denoise / Edge enhance（你已有的）
# ----------------------------------------

def denoise_luma_linear(bgr_linear: np.ndarray, strength: float = 0.4) -> np.ndarray:
    M = np.array([[0.2126, 0.7152, 0.0722],
                  [-0.1146, -0.3854, 0.5],
                  [0.5, -0.4542, -0.0458]], dtype=np.float32)
    rgb_linear = bgr_linear[..., ::-1]
    ycbcr = rgb_linear @ M.T
    Y = ycbcr[..., 0].astype(np.float32)
    Yf = cv2.bilateralFilter(
        Y,
        d=3,
        sigmaColor=0.05*(1+strength),
        sigmaSpace=1+int(1*strength),
    )
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

# ----------------------------------------
# Reconstruction 專用狀態
# ----------------------------------------

@dataclass
class ReconImageState:
    wb_gains: np.ndarray | None = None        # 給 WB temporal 用
    log_luma_ema: float | None = None         # 曝光穩定用
    prev_denoised_lin: np.ndarray | None = None  # temporal denoise 用

# ----------------------------------------
# 曝光穩定（在 linear luma 上做 global gain）
# ----------------------------------------

def stabilize_exposure_linear(
    bgr_linear: np.ndarray,
    state: ReconImageState,
    target_log_luma: float = -1.0,
    alpha: float = 0.9,
) -> tuple[np.ndarray, ReconImageState]:
    """
    在 linear 空間用 global gain 調整曝光，讓 log-luma 穩定在 target_log_luma。
    """
    B, G, R = bgr_linear[..., 0], bgr_linear[..., 1], bgr_linear[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    eps = 1e-6
    log_L = np.log(L + eps)
    cur_log_luma = float(np.mean(log_L))

    if state.log_luma_ema is None:
        state.log_luma_ema = cur_log_luma
    else:
        state.log_luma_ema = alpha * state.log_luma_ema + (1.0 - alpha) * cur_log_luma

    gain = np.exp(target_log_luma - state.log_luma_ema)
    gain = float(np.clip(gain, 0.5, 2.0))  # 防止太暴力

    bgr_out = bgr_linear * gain
    return bgr_out, state

# ----------------------------------------
# Denoise + temporal smoothing（linear space）
# ----------------------------------------

def denoise_luma_linear_temporal(
    bgr_linear: np.ndarray,
    state: ReconImageState,
    strength: float = 0.4,
    temporal_alpha: float = 0.8,
    temporal_diff_thresh: float = 0.02,
) -> tuple[np.ndarray, ReconImageState]:
    """
    在你原本的 denoise_luma_linear 上加 temporal blend：
    只有在亮度差異很小的地方才跟前一幀混合。
    """
    dll = denoise_luma_linear(bgr_linear, strength=strength)

    if state.prev_denoised_lin is not None and state.prev_denoised_lin.shape == dll.shape:
        # 在 linear space 上比較 luma
        B, G, R = dll[..., 0], dll[..., 1], dll[..., 2]
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B

        Bp, Gp, Rp = state.prev_denoised_lin[..., 0], state.prev_denoised_lin[..., 1], state.prev_denoised_lin[..., 2]
        Lp = 0.2126 * Rp + 0.7152 * Gp + 0.0722 * Bp

        diff = np.abs(L - Lp)
        mask_static = (diff < temporal_diff_thresh).astype(np.float32)[..., None]  # (H,W,1)

        dll = temporal_alpha * state.prev_denoised_lin * mask_static + dll * (1.0 - temporal_alpha * mask_static)

    state.prev_denoised_lin = dll.copy()
    return dll, state

# ----------------------------------------
# Pipeline：for 3D Reconstruction
# ----------------------------------------

def image_process_for_reconstruction(
    frame_bgr: np.ndarray,
    state: ReconImageState,
    wb_p: int = 6,
    wb_thresh: float = 0.0,
    wb_alpha: float = 0.8,
    denoise_strength: float = 0.4,
    do_sharpen: bool = True,
    use_ccm: bool = False,
) -> tuple[np.ndarray, np.ndarray, ReconImageState]:
    """
    輸入:
      - frame_bgr: USB camera 讀到的原始 BGR (uint8, sRGB)
      - state: ReconImageState (保存 WB/曝光/temporal denoise 狀態)

    輸出:
      - vis_bgr: 給顯示 / debug 用的 BGR uint8 (sRGB)
      - gray_for_slam: 給 feature / SLAM / stereo 的灰階 uint8
      - state: 更新後的 ReconImageState
    """

    # 1) WB (linear) - 用你原本的 temporal 版本
    wb_lin, gains = white_balance_sog_temporal(
        frame_bgr,
        p=wb_p,
        thresh=wb_thresh,
        prev_gains=state.wb_gains,
        alpha=wb_alpha,
    )
    state.wb_gains = gains

    # 2) linear 曝光穩定
    wb_lin, state = stabilize_exposure_linear(
        wb_lin,
        state=state,
        target_log_luma=-1.0,  # 之後可以改成初始化幾幀自動估
        alpha=0.9,
    )

    # 3) denoise (linear + temporal)
    dll, state = denoise_luma_linear_temporal(
        wb_lin,
        state=state,
        strength=denoise_strength,
        temporal_alpha=0.8,
        temporal_diff_thresh=0.02,
    )

    # 4) edge enhance in luma (optional)
    if do_sharpen:
        dll = enhance_edges_luma_linear(
            dll,
            amount=0.25,
            radius=2,
            thresh=0.02,
        )

    # 5) color correction CCM（可選）
    if use_ccm:
        dll = color_correction_ccm(dll)

    # 6) clamp + linear -> sRGB -> uint8
    dll = np.clip(dll, 0.0, 1.0).astype(np.float32)
    bgr_srgb = _linear_to_srgb(dll)
    bgr_srgb = np.clip(bgr_srgb * 255.0, 0, 255).astype(np.uint8)

    # 7) 產生給 SLAM / stereo 用的灰階
    gray_for_slam = cv2.cvtColor(bgr_srgb, cv2.COLOR_BGR2GRAY)

    return bgr_srgb, gray_for_slam, state

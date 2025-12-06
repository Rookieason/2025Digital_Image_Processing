import cv2
import numpy as np
from collections import deque
from typing import Optional


class HistogramMotionDetector:
    def __init__(self, n_bins: int = 32):
        self.n_bins = n_bins
        self._prev_hist: Optional[np.ndarray] = None

    def detect_motion(self, gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [self.n_bins], [0, 256])
        hist = hist.flatten()
        s = hist.sum()
        if s > 0:
            hist /= s

        if self._prev_hist is None:
            self._prev_hist = hist
            return 0.0

        num = (hist - self._prev_hist) ** 2
        den = hist + self._prev_hist + 1e-10
        chi2 = float(np.sum(num / den))

        self._prev_hist = hist
        motion_score = np.clip(chi2 * 10.0, 0.0, 1.0)
        return motion_score


class SLAMExposureCompensator:
    def __init__(
        self,
        target_percentile: float = 50.0,
        target_value: float = 128.0,
        min_gain: float = 0.7,
        max_gain: float = 1.4,
        base_adaptation_rate: float = 0.10,
        scene_change_alpha: float = 0.30,
        history_length: int = 10,
        scene_change_mad_threshold: float = 4.0,
        max_gain_step: float = 0.12,
        motion_step_scale: float = 0.5,
        gradient_downsample_scale: float = 0.25,
        gradient_blur_size: int = 3,
        edge_contrast_preservation: float = 0.5,
    ):
        self.target_percentile = float(target_percentile)
        self.target_value = float(target_value)

        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)

        self.base_adaptation_rate = float(base_adaptation_rate)
        self.scene_change_alpha = float(scene_change_alpha)

        self.history_length = int(history_length)
        self.scene_change_mad_threshold = float(scene_change_mad_threshold)

        self.max_gain_step = float(max_gain_step)
        self.motion_step_scale = float(motion_step_scale)

        self.gradient_downsample_scale = float(gradient_downsample_scale)
        self.gradient_blur_size = int(gradient_blur_size)
        self.edge_contrast_preservation = float(edge_contrast_preservation)

        self._intensity_history = deque(maxlen=self.history_length)
        self._current_gain = 1.0
        self._motion_detector = HistogramMotionDetector(n_bins=32)

        self._last_gradient_norm: Optional[np.ndarray] = None

    @staticmethod
    def _to_gray_bt709(frame_bgr: np.ndarray) -> np.ndarray:
        b = frame_bgr[..., 0].astype(np.float32)
        g = frame_bgr[..., 1].astype(np.float32)
        r = frame_bgr[..., 2].astype(np.float32)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.clip(y, 0, 255).astype(np.uint8)

    def _compute_robust_intensity(self, gray: np.ndarray) -> float:
        flat = gray.ravel()
        mask = (flat > 5) & (flat < 250)
        valid = flat[mask]
        if valid.size < 200:
            valid = flat
        intensity = np.percentile(valid, self.target_percentile)
        return float(intensity)

    def _detect_scene_change(self, current_intensity: float, motion_score: float) -> bool:
        if len(self._intensity_history) < 3:
            return False

        if motion_score > 0.25:
            return False

        hist = np.array(self._intensity_history, dtype=np.float32)
        median = np.median(hist)
        mad = np.median(np.abs(hist - median))
        threshold = self.scene_change_mad_threshold * (mad + 1e-6)
        return abs(current_intensity - median) > threshold

    def _compute_adaptive_gain(
        self,
        current_intensity: float,
        motion_score: float,
        scene_changed: bool,
    ) -> float:
        if current_intensity < 1e-3:
            target_gain = self.max_gain
        else:
            target_gain = self.target_value / current_intensity
        target_gain = float(np.clip(target_gain, self.min_gain, self.max_gain))

        if scene_changed:
            alpha = self.scene_change_alpha
        else:
            motion_penalty = 0.7 * np.clip(motion_score, 0.0, 1.0)
            alpha = self.base_adaptation_rate * (1.0 - motion_penalty)
            alpha = float(np.clip(alpha, 0.03, 0.20))

        raw_gain = alpha * target_gain + (1.0 - alpha) * self._current_gain

        step_limit = self.max_gain_step * (1.0 - self.motion_step_scale * np.clip(motion_score, 0.0, 1.0))
        step_limit = float(max(step_limit, 0.02))

        delta = raw_gain - self._current_gain
        if abs(delta) > step_limit:
            gain = self._current_gain + np.sign(delta) * step_limit
        else:
            gain = raw_gain

        return float(gain)

    def _compute_gradient_map_fast(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        scale = self.gradient_downsample_scale
        small_w = max(2, int(w * scale))
        small_h = max(2, int(h * scale))

        gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray_f = gray_small.astype(np.float32)

        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)

        gmax = float(grad_mag.max())
        if gmax < 1e-6:
            grad_norm_small = np.zeros_like(grad_mag, dtype=np.float32)
        else:
            grad_norm_small = (grad_mag / gmax).astype(np.float32)

        grad_norm = cv2.resize(grad_norm_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return grad_norm.astype(np.float32)

    def _build_contrast_preserving_gain_map(self, global_gain: float, grad_norm: np.ndarray) -> np.ndarray:
        strength = float(self.edge_contrast_preservation)
        exponent = 1.0 - strength * grad_norm
        exponent = np.clip(exponent, 0.5, 1.0).astype(np.float32)

        local_gain = np.power(global_gain, exponent).astype(np.float32)

        ksize = self.gradient_blur_size
        if ksize > 1:
            local_gain = cv2.GaussianBlur(local_gain, (ksize, ksize), 0)

        return local_gain

    def _apply_gain(self, frame_bgr: np.ndarray, gain_map: np.ndarray) -> np.ndarray:
        gm = gain_map[:, :, None]
        out = frame_bgr.astype(np.float32) * gm
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        assert frame_bgr.dtype == np.uint8

        gray = self._to_gray_bt709(frame_bgr)
        current_intensity = self._compute_robust_intensity(gray)

        motion_score = self._motion_detector.detect_motion(gray)

        self._intensity_history.append(current_intensity)
        scene_changed = self._detect_scene_change(current_intensity, motion_score)

        if scene_changed:
            self._intensity_history.clear()
            self._intensity_history.append(current_intensity)

        gain = self._compute_adaptive_gain(current_intensity, motion_score, scene_changed)
        self._current_gain = gain

        if abs(gain - 1.0) < 0.01:
            return frame_bgr

        grad_norm = self._compute_gradient_map_fast(gray)
        self._last_gradient_norm = grad_norm

        gain_map = self._build_contrast_preserving_gain_map(gain, grad_norm)
        corrected = self._apply_gain(frame_bgr, gain_map)
        return corrected

    def get_diagnostics(self) -> dict:
        hist = list(self._intensity_history)
        return {
            "current_gain": float(self._current_gain),
            "intensity_history": hist,
            "history_std": float(np.std(hist)) if hist else 0.0,
        }

    def get_gradient_visualization(self) -> Optional[np.ndarray]:
        if self._last_gradient_norm is None:
            return None
        return (self._last_gradient_norm * 255.0).astype(np.uint8)


def integrate_into_pipeline(pipeline_instance, compensator: Optional[SLAMExposureCompensator] = None):
    if compensator is None:
        compensator = SLAMExposureCompensator()

    def new_exposure_compensation(self, img_u8: np.ndarray) -> np.ndarray:
        return compensator.process(img_u8)

    pipeline_instance.exposure_compensation = new_exposure_compensation.__get__(
        pipeline_instance, type(pipeline_instance)
    )
    return pipeline_instance

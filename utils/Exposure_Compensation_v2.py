import cv2
import numpy as np
from collections import deque


class MotionAwareFilter:
    """
    Simple motion detector based on frame difference.
    Output: motion score in [0, 1] (0 = static, 1 = strong motion).
    """
    def __init__(self):
        self._prev_gray = None

    def detect_motion(self, gray: np.ndarray) -> float:
        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return 0.0

        diff = cv2.absdiff(gray, self._prev_gray)
        motion_score = float(diff.mean()) / 255.0
        self._prev_gray = gray.copy()
        return np.clip(motion_score, 0.0, 1.0)


class SLAMExposureCompensator:
    """
    Exposure compensation optimized for real-time USB camera SLAM (improved v2).

    目標（針對 SLAM / Direct VO 實際問題）:
    ------------------------------------
    1. 時間穩定：抑制曝光跳動，避免 tracking 因亮度變化崩掉
    2. 對突變敏感：場景大改變時能快速調整，不拖泥帶水
    3. 對 outlier robust：不被高光 / 大暗區 / 飽和區嚴重誤導
    4. 保護梯度：曝光補償不破壞 edge / corner 的對比（Direct 法用得到）
    5. 計算成本合理：灰階 + 梯度只算一次，三個 channel 共用

    改動重點相對舊版：
    -------------------
    - 亮度估計：用 BT.709 灰階 + 排除飽和/極暗像素 → robust percentile
    - 場景穩定度：改用 temporal motion（frame diff），不再用 3x3 CV 當 proxy
    - 場景切換：MAD 檢測後會 reset 歷史，快速建立新 baseline
    - 梯度保護：在灰階算一次 Sobel，連續權重 w(grad)，非硬 threshold
    - 拿掉沒實質作用的 reference_intensity 結構，state 更乾淨
    """

    def __init__(
        self,
        target_percentile: float = 50.0,       # 用哪個 percentile 當場景亮度代表（50% ≈ median）
        target_value: float = 128.0,           # 希望該 percentile 對應的亮度（中灰）
        base_adaptation_rate: float = 0.15,    # 基本適應速度（建議 0.1~0.3）
        max_gain: float = 2.5,                 # 最大提亮倍數
        min_gain: float = 0.4,                 # 最大壓暗倍數
        history_length: int = 10,              # 亮度歷史長度（用於 MAD）
        scene_change_mad_mul: float = 3.0,     # 幾倍 MAD 視為場景變化
        scene_change_min_alpha: float = 0.35,  # 場景變化時最小 alpha（加快收斂）
        min_stability: float = 0.3,            # motion 再大，alpha 至少保留的比例
        gradient_threshold: float = 0.3,       # 梯度強度開始保護的門檻（0~1）
        edge_protection_strength: float = 0.2  # 邊緣區保留多少 global gain（0~1，越小保護越強）
    ):
        self.target_percentile = float(target_percentile)
        self.target_value = float(target_value)
        self.base_adaptation_rate = float(base_adaptation_rate)
        self.max_gain = float(max_gain)
        self.min_gain = float(min_gain)
        self.history_length = int(history_length)
        self.scene_change_mad_mul = float(scene_change_mad_mul)
        self.scene_change_min_alpha = float(scene_change_min_alpha)
        self.min_stability = float(min_stability)
        self.gradient_threshold = float(gradient_threshold)
        self.edge_protection_strength = float(edge_protection_strength)

        # Internal state
        self._intensity_history = deque(maxlen=self.history_length)
        self._current_gain = 1.0
        self._motion_filter = MotionAwareFilter()

    # ------------------------------------------------------------------
    # 灰階 / brightness estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray_bt709(frame_bgr: np.ndarray) -> np.ndarray:
        """
        BT.709 luminance (與 pipeline 最終灰階一致)。
        回傳 uint8 [0, 255]。
        """
        b = frame_bgr[:, :, 0].astype(np.float32)
        g = frame_bgr[:, :, 1].astype(np.float32)
        r = frame_bgr[:, :, 2].astype(np.float32)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        y_u8 = np.clip(y, 0.0, 255.0).astype(np.uint8)
        return y_u8

    def _compute_robust_intensity(self, gray_u8: np.ndarray) -> float:
        """
        使用 percentile + 排除極暗/飽和像素的 robust brightness。
        """
        gray = gray_u8.reshape(-1)
        # 排除過暗 / 過亮（接近 0/255 的極值）
        mask = (gray > 5) & (gray < 250)
        valid = gray[mask]
        if valid.size < 500:  # 幾乎整張都是極端值，退回用全部像素
            valid = gray
        intensity = np.percentile(valid, self.target_percentile)
        return float(intensity)

    # ------------------------------------------------------------------
    # Scene change & motion
    # ------------------------------------------------------------------
    def _detect_scene_change(self, current_intensity: float) -> bool:
        """
        用 MAD 判斷亮度統計是否突然跳動 → 場景切換。
        """
        if len(self._intensity_history) < 3:
            return False

        history_array = np.array(self._intensity_history, dtype=np.float32)
        median = np.median(history_array)
        mad = np.median(np.abs(history_array - median))
        threshold = self.scene_change_mad_mul * (mad + 1e-6)
        return abs(current_intensity - median) > threshold

    def _update_history_on_scene_change(self, current_intensity: float):
        """
        場景切換時，重設 history，以免舊場景拖累判斷。
        """
        self._intensity_history.clear()
        self._intensity_history.append(current_intensity)

    # ------------------------------------------------------------------
    # Gain computation
    # ------------------------------------------------------------------
    def _compute_gain(
        self,
        current_intensity: float,
        motion_score: float,
        scene_changed: bool
    ) -> float:
        """
        依照亮度 / 運動 / 場景變化來決定新的曝光 gain。
        """
        if current_intensity < 1e-3:
            target_gain = self.max_gain
        else:
            target_gain = self.target_value / current_intensity

        target_gain = float(np.clip(target_gain, self.min_gain, self.max_gain))

        # motion_score ∈ [0,1]，動越大 → 稍微降低 adaptation rate
        stability = 1.0 - 0.7 * np.clip(motion_score, 0.0, 1.0)
        stability = float(np.clip(stability, self.min_stability, 1.0))

        alpha = self.base_adaptation_rate * stability

        if scene_changed:
            # 場景換掉時，加快收斂，但不超過 0.5 以防 overshoot
            alpha = max(alpha, self.scene_change_min_alpha)
            alpha = min(alpha, 0.5)

        gain = alpha * target_gain + (1.0 - alpha) * self._current_gain
        return float(gain)

    # ------------------------------------------------------------------
    # Gradient-preserving compensation
    # ------------------------------------------------------------------
    def _compute_gradient_norm(self, gray_u8: np.ndarray) -> np.ndarray:
        """
        在灰階上算 Sobel gradient，回傳 [0,1] 的 norm。
        """
        gray_f = gray_u8.astype(np.float32)
        grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gmax = float(grad_mag.max())
        if gmax < 1e-6:
            return np.zeros_like(gray_f, dtype=np.float32)
        grad_norm = grad_mag / gmax
        return grad_norm.astype(np.float32)

    def _build_local_gain_map(self, gain: float, grad_norm: np.ndarray) -> np.ndarray:
        """
        連續權重的 local gain map：
        - gradient 小 → 接近 global gain
        - gradient 大 → 接近「受保護的 edge gain」
        """
        # 將梯度值映射到 [0,1] 的權重 w：
        #   grad < threshold → w=0
        #   grad → 1 → w→1
        th = self.gradient_threshold
        w = (grad_norm - th) / max(1e-6, (1.0 - th))
        w = np.clip(w, 0.0, 1.0).astype(np.float32)

        # 邊緣區的 gain：只施加部分 global gain（保護梯度）
        # edge_gain = 1 + (gain - 1) * edge_protection_strength
        edge_gain = 1.0 + (gain - 1.0) * float(self.edge_protection_strength)

        # local_gain = (1-w) * gain + w * edge_gain
        local_gain = (1.0 - w) * gain + w * edge_gain

        # 為避免局部不連續，做一次 GaussianBlur
        local_gain = cv2.GaussianBlur(local_gain, (5, 5), 0).astype(np.float32)
        return local_gain

    def _apply_gain_map(self, frame_bgr: np.ndarray, gain_map: np.ndarray) -> np.ndarray:
        """
        對 BGR 每個 channel 套用同一個 local gain map。
        gain_map: float32, shape (H, W)，已平滑。
        """
        img_f = frame_bgr.astype(np.float32)
        gm = gain_map[:, :, None]  # (H,W) -> (H,W,1) for broadcasting
        corrected = img_f * gm
        corrected = np.clip(corrected, 0.0, 255.0).astype(np.uint8)
        return corrected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Apply SLAM-oriented exposure compensation.

        Input : uint8 BGR frame
        Output: exposure-compensated uint8 BGR frame
        """
        assert frame_bgr.dtype == np.uint8, "SLAMExposureCompensator expects uint8 BGR input."

        # 1) 灰階 (BT.709) for analysis
        gray = self._to_gray_bt709(frame_bgr)

        # 2) robust brightness
        current_intensity = self._compute_robust_intensity(gray)

        # 3) motion detection
        motion_score = self._motion_filter.detect_motion(gray)

        # 4) update history & detect scene change
        self._intensity_history.append(current_intensity)
        scene_changed = self._detect_scene_change(current_intensity)

        if scene_changed:
            self._update_history_on_scene_change(current_intensity)

        # 5) compute gain
        gain = self._compute_gain(current_intensity, motion_score, scene_changed)
        self._current_gain = gain

        if abs(gain - 1.0) < 1e-3:
            # 近似無需補償，直接返回
            return frame_bgr

        # 6) gradient-aware local gain map
        grad_norm = self._compute_gradient_norm(gray)
        gain_map = self._build_local_gain_map(gain, grad_norm)

        # 7) apply local gain map to BGR
        corrected = self._apply_gain_map(frame_bgr, gain_map)
        return corrected

    def get_diagnostics(self) -> dict:
        """
        用於 debug / 可視化：
        回傳目前 gain、亮度歷史分布、motion 等資訊。
        """
        hist = list(self._intensity_history)
        return {
            "current_gain": float(self._current_gain),
            "intensity_history": hist,
            "history_std": float(np.std(hist)) if hist else 0.0,
            "history_len": len(hist),
        }


# ============================================================================
# Integration: drop-in replacement for original exposure_compensation
# ============================================================================

def integrate_into_pipeline(pipeline_instance, compensator: SLAMExposureCompensator | None = None):
    """
    將改良後的 SLAMExposureCompensator 掛進 USBBaselinePipelineFast。

    用法：
        pipeline = USBBaselinePipelineFast(...)
        pipeline = integrate_into_pipeline(pipeline)
        # 之後 pipeline.process(frame) 就會用新的曝光補償

    也可以自行建立 compensator 丟進來以調整參數：
        compensator = SLAMExposureCompensator(target_value=140, base_adaptation_rate=0.2)
        pipeline = integrate_into_pipeline(pipeline, compensator)
    """
    if compensator is None:
        compensator = SLAMExposureCompensator(
            target_percentile=50.0,
            target_value=128.0,
            base_adaptation_rate=0.15,
            max_gain=2.5,
            min_gain=0.4,
            history_length=10,
            gradient_threshold=0.3,
            edge_protection_strength=0.2,
        )

    def new_exposure_compensation(self, img_u8: np.ndarray) -> np.ndarray:
        return compensator.process(img_u8)

    pipeline_instance.exposure_compensation = new_exposure_compensation.__get__(
        pipeline_instance, type(pipeline_instance)
    )
    return pipeline_instance

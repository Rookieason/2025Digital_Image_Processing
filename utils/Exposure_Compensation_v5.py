import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExposureMetrics:
    """Metrics for reconstruction quality assessment"""
    current_gain: float
    intensity_mean: float
    intensity_std: float
    information_content: float  # Entropy-based
    temporal_stability: float
    saturation_ratio: float
    motion_score: float


class HistogramMotionDetector:
    """Fast histogram-based motion detection"""
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

        # Chi-squared distance
        num = (hist - self._prev_hist) ** 2
        den = hist + self._prev_hist + 1e-10
        chi2 = float(np.sum(num / den))

        self._prev_hist = hist
        motion_score = np.clip(chi2 * 10.0, 0.0, 1.0)
        return motion_score

    def reset(self):
        """Reset motion detection state"""
        self._prev_hist = None


class SLAMExposureCompensator:
    """
    Exposure compensation optimized for Direct SLAM reconstruction.
    
    Key improvements over standard approaches:
    1. Photometric consistency preservation
    2. Temporal smoothness for tracking
    3. Information content maximization
    4. Saturation avoidance (preserves texture for reconstruction)
    5. Minimal local adaptation (global is better for SLAM)
    """
    
    def __init__(
        self,
        # Target intensity settings
        target_percentile: float = 50.0,
        target_value: float = 128.0,
        
        # Gain limits (conservative for SLAM)
        min_gain: float = 0.75,
        max_gain: float = 1.35,
        
        # Adaptation rates
        base_adaptation_rate: float = 0.08,  # Slower for stability
        fast_adaptation_rate: float = 0.25,  # For scene changes
        
        # Temporal smoothing
        temporal_smoothing_factor: float = 0.15,  # Smooth gain transitions
        gain_acceleration_limit: float = 0.05,  # Limit jerk
        
        # History and scene detection
        history_length: int = 15,
        scene_change_threshold: float = 5.0,  # MAD multiplier
        
        # Saturation handling
        saturation_threshold: float = 0.05,  # Max 5% saturated pixels
        saturation_penalty: float = 0.7,     # Reduce gain if saturating
        
        # Information content
        target_entropy_ratio: float = 0.70,  # Target histogram spread
        
        # Local adaptation (minimal for SLAM)
        use_local_adaptation: bool = False,
        local_adaptation_strength: float = 0.2,
        
        # Motion handling
        motion_freeze_threshold: float = 0.4,  # Freeze updates during fast motion
    ):
        # Basic parameters
        self.target_percentile = float(target_percentile)
        self.target_value = float(target_value)
        
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        
        self.base_adaptation_rate = float(base_adaptation_rate)
        self.fast_adaptation_rate = float(fast_adaptation_rate)
        
        self.temporal_smoothing_factor = float(temporal_smoothing_factor)
        self.gain_acceleration_limit = float(gain_acceleration_limit)
        
        self.history_length = int(history_length)
        self.scene_change_threshold = float(scene_change_threshold)
        
        self.saturation_threshold = float(saturation_threshold)
        self.saturation_penalty = float(saturation_penalty)
        
        self.target_entropy_ratio = float(target_entropy_ratio)
        
        self.use_local_adaptation = bool(use_local_adaptation)
        self.local_adaptation_strength = float(local_adaptation_strength)
        
        self.motion_freeze_threshold = float(motion_freeze_threshold)
        
        # State
        self._intensity_history = deque(maxlen=self.history_length)
        self._gain_history = deque(maxlen=3)  # For acceleration limiting
        self._current_gain = 1.0
        self._target_gain = 1.0
        
        self._motion_detector = HistogramMotionDetector(n_bins=32)
        
        self._frame_count = 0
        self._scene_stable_frames = 0
        
    @staticmethod
    def _to_gray_bt709(frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale using BT.709 standard"""
        b = frame_bgr[..., 0].astype(np.float32)
        g = frame_bgr[..., 1].astype(np.float32)
        r = frame_bgr[..., 2].astype(np.float32)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.clip(y, 0, 255).astype(np.uint8)
    
    def _compute_robust_statistics(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute robust intensity statistics avoiding saturation regions.
        Returns: (intensity, std, saturation_ratio)
        """
        flat = gray.ravel()
        
        # Compute saturation ratio (critical for reconstruction)
        n_total = flat.size
        n_saturated = np.sum((flat <= 5) | (flat >= 250))
        saturation_ratio = float(n_saturated / n_total)
        
        # Use middle range for statistics
        mask = (flat > 10) & (flat < 245)
        valid = flat[mask]
        
        if valid.size < 100:
            valid = flat
        
        intensity = np.percentile(valid, self.target_percentile)
        std = np.std(valid)
        
        return float(intensity), float(std), saturation_ratio
    
    def _compute_information_content(self, gray: np.ndarray) -> float:
        """
        Compute normalized entropy as measure of information content.
        Higher entropy = more texture detail for reconstruction.
        """
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove zero bins
        
        if hist.size == 0:
            return 0.0
        
        # Normalize
        hist = hist / hist.sum()
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalize to [0, 1] (max entropy for 64 bins is log2(64) = 6)
        normalized_entropy = entropy / 6.0
        
        return float(np.clip(normalized_entropy, 0.0, 1.0))
    
    def _detect_scene_change(
        self, 
        current_intensity: float, 
        motion_score: float
    ) -> bool:
        """
        Detect scene changes using MAD (Median Absolute Deviation).
        More conservative than previous version.
        """
        if len(self._intensity_history) < 5:
            return False
        
        # Don't detect scene changes during motion
        if motion_score > 0.3:
            return False
        
        hist = np.array(self._intensity_history, dtype=np.float32)
        median = np.median(hist)
        mad = np.median(np.abs(hist - median))
        
        threshold = self.scene_change_threshold * (mad + 5.0)  # Add offset
        deviation = abs(current_intensity - median)
        
        return deviation > threshold
    
    def _compute_target_gain(
        self,
        current_intensity: float,
        saturation_ratio: float,
        information_content: float,
    ) -> float:
        """
        Compute target gain considering multiple factors.
        """
        # Basic intensity-based target
        if current_intensity < 1e-3:
            target = self.max_gain
        else:
            target = self.target_value / current_intensity
        
        # Penalize if too much saturation
        if saturation_ratio > self.saturation_threshold:
            penalty = self.saturation_penalty ** ((saturation_ratio - self.saturation_threshold) * 10)
            target = min(target, self._current_gain * penalty)
        
        # Encourage information content (but gently)
        if information_content < self.target_entropy_ratio:
            # Slightly increase gain to reveal more detail
            entropy_boost = 1.0 + 0.1 * (self.target_entropy_ratio - information_content)
            target *= entropy_boost
        
        # Clamp to safe range
        target = float(np.clip(target, self.min_gain, self.max_gain))
        
        return target
    
    def _smooth_gain_transition(
        self,
        target_gain: float,
        motion_score: float,
        scene_changed: bool,
    ) -> float:
        """
        Apply temporal smoothing with acceleration limiting.
        Critical for SLAM tracking stability.
        """
        # Freeze during very fast motion
        if motion_score > self.motion_freeze_threshold:
            return self._current_gain
        
        # Choose adaptation rate
        if scene_changed:
            alpha = self.fast_adaptation_rate
        else:
            # Slower during motion
            motion_penalty = 0.5 * np.clip(motion_score, 0.0, 1.0)
            alpha = self.base_adaptation_rate * (1.0 - motion_penalty)
            alpha = float(np.clip(alpha, 0.03, self.base_adaptation_rate))
        
        # Exponential moving average
        smooth_gain = alpha * target_gain + (1.0 - alpha) * self._current_gain
        
        # Limit acceleration (jerk control)
        if len(self._gain_history) >= 2:
            prev_velocity = self._gain_history[-1] - self._gain_history[-2]
            new_velocity = smooth_gain - self._current_gain
            acceleration = new_velocity - prev_velocity
            
            if abs(acceleration) > self.gain_acceleration_limit:
                # Limit acceleration
                limited_velocity = prev_velocity + np.sign(acceleration) * self.gain_acceleration_limit
                smooth_gain = self._current_gain + limited_velocity
        
        # Additional temporal smoothing
        final_gain = (
            self.temporal_smoothing_factor * smooth_gain + 
            (1.0 - self.temporal_smoothing_factor) * self._current_gain
        )
        
        return float(np.clip(final_gain, self.min_gain, self.max_gain))
    
    def _apply_global_gain(
        self, 
        frame_bgr: np.ndarray, 
        gain: float
    ) -> np.ndarray:
        """Apply global gain (best for SLAM photometric consistency)"""
        out = frame_bgr.astype(np.float32) * gain
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)
    
    def _apply_local_gain(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        global_gain: float,
    ) -> np.ndarray:
        """
        Apply mild local adaptation to preserve highlights/shadows.
        Use sparingly - can hurt photometric consistency.
        """
        # Compute local mean using fast box filter
        ksize = 31
        local_mean = cv2.blur(gray.astype(np.float32), (ksize, ksize))
        
        # Compute deviation from target
        deviation = local_mean - self.target_value
        
        # Create local gain adjustment (mild)
        local_adjustment = 1.0 - self.local_adaptation_strength * (deviation / 128.0)
        local_adjustment = np.clip(local_adjustment, 0.7, 1.3).astype(np.float32)
        
        # Combine with global gain
        combined_gain = global_gain * local_adjustment
        
        # Apply
        out = frame_bgr.astype(np.float32) * combined_gain[:, :, None]
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)
    
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Process frame with exposure compensation.
        
        Args:
            frame_bgr: Input BGR frame (uint8)
            
        Returns:
            Compensated BGR frame (uint8)
        """
        assert frame_bgr.dtype == np.uint8
        self._frame_count += 1
        
        # Convert to grayscale
        gray = self._to_gray_bt709(frame_bgr)
        
        # Compute statistics
        current_intensity, intensity_std, saturation_ratio = self._compute_robust_statistics(gray)
        information_content = self._compute_information_content(gray)
        
        # Detect motion
        motion_score = self._motion_detector.detect_motion(gray)
        
        # Update history
        self._intensity_history.append(current_intensity)
        
        # Detect scene changes
        scene_changed = self._detect_scene_change(current_intensity, motion_score)
        
        if scene_changed:
            # Reset for new scene
            self._intensity_history.clear()
            self._intensity_history.append(current_intensity)
            self._gain_history.clear()
            self._scene_stable_frames = 0
            self._motion_detector.reset()
        else:
            self._scene_stable_frames += 1
        
        # Compute target gain
        target_gain = self._compute_target_gain(
            current_intensity,
            saturation_ratio,
            information_content,
        )
        self._target_gain = target_gain
        
        # Smooth gain transition
        gain = self._smooth_gain_transition(
            target_gain,
            motion_score,
            scene_changed,
        )
        
        # Update state
        self._gain_history.append(gain)
        self._current_gain = gain
        
        # Early exit if no correction needed
        if abs(gain - 1.0) < 0.02:
            return frame_bgr
        
        # Apply gain
        if self.use_local_adaptation and self._scene_stable_frames > 10:
            # Only use local adaptation after scene is stable
            corrected = self._apply_local_gain(frame_bgr, gray, gain)
        else:
            # Global gain (preferred for SLAM)
            corrected = self._apply_global_gain(frame_bgr, gain)
        
        return corrected
    
    def get_metrics(self) -> ExposureMetrics:
        """Get current exposure metrics for monitoring"""
        hist = list(self._intensity_history)
        
        if len(hist) > 1:
            temporal_stability = 1.0 - (np.std(hist) / 128.0)
        else:
            temporal_stability = 1.0
        
        return ExposureMetrics(
            current_gain=float(self._current_gain),
            intensity_mean=float(np.mean(hist)) if hist else 128.0,
            intensity_std=float(np.std(hist)) if hist else 0.0,
            information_content=0.0,  # Updated during process
            temporal_stability=float(np.clip(temporal_stability, 0.0, 1.0)),
            saturation_ratio=0.0,  # Updated during process
            motion_score=0.0,  # Updated during process
        )
    
    def reset(self):
        """Reset all state"""
        self._intensity_history.clear()
        self._gain_history.clear()
        self._current_gain = 1.0
        self._target_gain = 1.0
        self._motion_detector.reset()
        self._frame_count = 0
        self._scene_stable_frames = 0


def integrate_into_pipeline(
    pipeline_instance, 
    compensator: Optional[SLAMExposureCompensator] = None
):
    """
    Integrate compensator into existing pipeline.
    
    Usage:
        compensator = SLAMExposureCompensator(
            use_local_adaptation=False,  # Recommended for SLAM
            base_adaptation_rate=0.08,    # Slower for stability
        )
        pipeline = integrate_into_pipeline(pipeline, compensator)
    """
    if compensator is None:
        compensator = SLAMExposureCompensator()
    
    def new_exposure_compensation(self, img_u8: np.ndarray) -> np.ndarray:
        return compensator.process(img_u8)
    
    pipeline_instance.exposure_compensation = new_exposure_compensation.__get__(
        pipeline_instance, type(pipeline_instance)
    )
    
    return pipeline_instance

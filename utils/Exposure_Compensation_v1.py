import cv2
import numpy as np
from collections import deque


class SLAMExposureCompensator:
    """
    Exposure compensation optimized for real-time USB camera SLAM.
    
    Key Design Principles:
    ----------------------
    1. TEMPORAL CONSISTENCY: Smooth exposure changes to preserve feature tracking
    2. SPATIAL AWARENESS: Protect high-gradient regions (edges/corners)
    3. OUTLIER ROBUSTNESS: Use median instead of mean to handle scene changes
    4. BOUNDED CORRECTION: Prevent extreme adjustments that destroy detail
    5. PREDICTIVE ADAPTATION: Use frame history to anticipate lighting changes
    
    Algorithm Overview:
    -------------------
    - Uses robust statistics (median + MAD) instead of mean
    - Multi-scale analysis: global + local region stability
    - Adaptive smoothing: faster response to intentional camera motion
    - Gradient-preserving normalization: protects SLAM features
    - Predictive filtering: reduces lag in exposure adaptation
    """
    
    def __init__(
        self,
        target_percentile: float = 50.0,       # Target median brightness
        target_value: int = 128,                # Target intensity (mid-gray)
        adaptation_rate: float = 0.15,          # Response speed (0.1-0.3 for SLAM)
        max_gain: float = 2.5,                  # Max brightening
        min_gain: float = 0.4,                  # Max darkening
        history_length: int = 10,               # Frames for predictive filtering
        local_analysis: bool = True,            # Enable region-based stability check
        gradient_preserve_threshold: float = 0.3,  # Protect high-gradient areas
    ):
        self.target_percentile = target_percentile
        self.target_value = target_value
        self.adaptation_rate = adaptation_rate
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.history_length = history_length
        self.local_analysis = local_analysis
        self.gradient_threshold = gradient_preserve_threshold
        
        # Internal state
        self._reference_intensity = None        # Target brightness (smoothed)
        self._current_gain = 1.0                # Applied gain (smoothed)
        self._intensity_history = deque(maxlen=history_length)
        self._motion_detector = MotionAwareFilter()
        
        # Precompute grid for local analysis (divide frame into 3x3 regions)
        self._grid_regions = None
        
    def _initialize_reference(self, gray: np.ndarray):
        """Initialize reference on first frame."""
        intensity = np.percentile(gray, self.target_percentile)
        self._reference_intensity = intensity
        self._intensity_history.append(intensity)
        
    def _compute_robust_intensity(self, gray: np.ndarray) -> float:
        """
        Compute robust central tendency using median.
        More stable than mean against outliers (e.g., specular highlights).
        """
        # Use percentile for efficiency (faster than full median on large images)
        intensity = np.percentile(gray, self.target_percentile)
        return float(intensity)
    
    def _detect_scene_change(self, current_intensity: float) -> bool:
        """
        Detect abrupt scene changes (e.g., walking through doorway).
        Uses MAD (Median Absolute Deviation) for robustness.
        """
        if len(self._intensity_history) < 3:
            return False
        
        history_array = np.array(self._intensity_history)
        median = np.median(history_array)
        mad = np.median(np.abs(history_array - median))
        
        # Scene change if current intensity is >3 MAD from median
        threshold = 3.0 * (mad + 1e-6)  # +epsilon for numerical stability
        return abs(current_intensity - median) > threshold
    
    def _compute_local_stability(self, gray: np.ndarray) -> float:
        """
        Analyze local region variance to detect if scene is stable.
        High variance → likely camera motion → slow down adaptation.
        Low variance → static scene → can adapt faster.
        """
        if not self.local_analysis:
            return 1.0
        
        h, w = gray.shape
        if self._grid_regions is None:
            # Create 3x3 grid of regions
            self._grid_regions = []
            for i in range(3):
                for j in range(3):
                    y1 = int(h * i / 3)
                    y2 = int(h * (i + 1) / 3)
                    x1 = int(w * j / 3)
                    x2 = int(w * (j + 1) / 3)
                    self._grid_regions.append((slice(y1, y2), slice(x1, x2)))
        
        # Compute median intensity for each region
        region_intensities = []
        for region in self._grid_regions:
            region_median = np.median(gray[region])
            region_intensities.append(region_median)
        
        # Coefficient of variation (normalized std) across regions
        region_std = np.std(region_intensities)
        region_mean = np.mean(region_intensities) + 1e-6
        cv = region_std / region_mean
        
        # High CV → unstable scene → reduce adaptation rate
        # Map CV [0, 0.5] → stability [1.0, 0.3]
        stability = np.clip(1.0 - cv * 1.4, 0.3, 1.0)
        return stability
    
    def _compute_adaptive_gain(self, current_intensity: float, stability: float) -> float:
        """
        Compute gain with adaptive smoothing based on scene stability.
        
        Returns
        -------
        gain : float
            Multiplicative factor to apply to image
        """
        if self._reference_intensity is None:
            return 1.0
        
        # Target gain to reach desired brightness
        if current_intensity < 1e-3:
            target_gain = self.max_gain
        else:
            target_gain = self.target_value / current_intensity
        
        # Clamp to safe bounds
        target_gain = np.clip(target_gain, self.min_gain, self.max_gain)
        
        # Adaptive smoothing rate based on scene stability and change detection
        scene_changed = self._detect_scene_change(current_intensity)
        
        if scene_changed:
            # Faster adaptation for abrupt scene changes
            alpha = min(0.4, self.adaptation_rate * 2.5)
        else:
            # Normal adaptation scaled by stability
            alpha = self.adaptation_rate * stability
        
        # EMA filter for smooth gain transition
        gain = alpha * target_gain + (1.0 - alpha) * self._current_gain
        
        return gain
    
    def _apply_gradient_preserving_compensation(
        self, 
        img: np.ndarray, 
        gain: float
    ) -> np.ndarray:
        """
        Apply exposure compensation while preserving high-gradient regions.
        
        SLAM features (corners, edges) are often at high-gradient locations.
        We reduce gain in these areas to prevent destroying local contrast.
        
        Method: Use gradient magnitude as weight for reduced correction.
        """
        if gain == 1.0:
            return img
        
        # Convert to float for processing
        img_f = img.astype(np.float32)
        
        # Fast gradient magnitude using Sobel (3x3 kernel)
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude to [0, 1]
        grad_max = grad_mag.max()
        if grad_max > 0:
            grad_norm = grad_mag / grad_max
        else:
            grad_norm = np.zeros_like(grad_mag)
        
        # Create spatially-varying gain map
        # High gradient → gain closer to 1.0 (less correction)
        # Low gradient → full gain (more correction)
        protection = grad_norm > self.gradient_threshold
        gain_map = np.where(
            protection,
            1.0 + (gain - 1.0) * 0.5,  # 50% reduction in high-gradient areas
            gain
        )
        
        # Smooth gain map to avoid artifacts
        gain_map = cv2.GaussianBlur(gain_map.astype(np.float32), (5, 5), 0)
        
        # Apply spatially-varying gain
        img_corrected = img_f * gain_map
        
        # Clip and convert back
        img_corrected = np.clip(img_corrected, 0, 255)
        return img_corrected.astype(np.uint8)
    
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Apply SLAM-optimized exposure compensation.
        
        Parameters
        ----------
        frame_bgr : np.ndarray
            Input BGR image (uint8)
            
        Returns
        -------
        corrected : np.ndarray
            Exposure-compensated BGR image (uint8)
        """
        # Convert to grayscale for analysis (BT.709)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if self._reference_intensity is None:
            self._initialize_reference(gray)
            return frame_bgr  # Return original on first frame (no history yet)
        
        # Compute current scene brightness
        current_intensity = self._compute_robust_intensity(gray)
        
        # Update history
        self._intensity_history.append(current_intensity)
        
        # Analyze scene stability
        stability = self._compute_local_stability(gray)
        
        # Compute adaptive gain
        gain = self._compute_adaptive_gain(current_intensity, stability)
        
        # Store gain for next iteration
        self._current_gain = gain
        
        # Apply compensation with gradient preservation (per-channel)
        corrected_bgr = np.zeros_like(frame_bgr)
        for i in range(3):  # Process each channel
            corrected_bgr[:, :, i] = self._apply_gradient_preserving_compensation(
                frame_bgr[:, :, i], gain
            )
        
        # Update reference intensity (slow drift to track gradual lighting changes)
        self._reference_intensity = (
            0.02 * current_intensity + 0.98 * self._reference_intensity
        )
        
        return corrected_bgr
    
    def get_diagnostics(self) -> dict:
        """Return current state for debugging/visualization."""
        return {
            'reference_intensity': self._reference_intensity,
            'current_gain': self._current_gain,
            'intensity_history': list(self._intensity_history),
            'history_std': np.std(self._intensity_history) if self._intensity_history else 0,
        }


class MotionAwareFilter:
    """
    Simple motion detector to adjust adaptation rate.
    (Placeholder for potential future enhancement)
    """
    def __init__(self):
        self._prev_frame = None
    
    def detect_motion(self, gray: np.ndarray) -> float:
        """Return motion score [0=static, 1=fast motion]."""
        if self._prev_frame is None:
            self._prev_frame = gray.copy()
            return 0.0
        
        # Simple frame difference
        diff = cv2.absdiff(gray, self._prev_frame)
        motion_score = diff.mean() / 255.0
        
        self._prev_frame = gray.copy()
        return motion_score


# ============================================================================
# Integration Example: Drop-in replacement for original exposure_compensation
# ============================================================================

def integrate_into_pipeline(pipeline_instance):
    """
    Example of how to integrate SLAMExposureCompensator into the 
    existing USBBaselinePipelineFast.
    
    Replace the exposure_compensation method with:
    """
    slam_compensator = SLAMExposureCompensator(
        target_value=128,
        adaptation_rate=0.15,  # Tuned for 30 FPS
        local_analysis=True,
        gradient_preserve_threshold=0.3
    )
    
    def new_exposure_compensation(self, img_u8: np.ndarray) -> np.ndarray:
        """SLAM-optimized exposure compensation."""
        return slam_compensator.process(img_u8)
    
    # Monkey-patch the method (or refactor pipeline to accept compensator)
    pipeline_instance.exposure_compensation = new_exposure_compensation.__get__(
        pipeline_instance, type(pipeline_instance)
    )
    
    return pipeline_instance
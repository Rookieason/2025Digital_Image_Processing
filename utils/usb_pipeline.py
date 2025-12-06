# usb_baseline_pipeline_fast.py
import cv2
import numpy as np


class USBBaselinePipelineFast:
    """
    USB Camera baseline pipeline (real-time friendly):

    1. Photometric Calibration → CRF LUT (if provided) or fixed gamma as fallback
    2. Vignetting Removal      → divide by precomputed radial mask (on distorted image)
    3. Exposure Compensation   → mean-luminance based global scaling
    4. Undistortion            → precomputed remap (initUndistortRectifyMap + remap)
    5. Grayscale Conversion    → BT.709 luminance
    6. Light Denoising         → Gaussian blur (3x3, fast); optional bilateral

    Notes:
    - If a CRF LUT is given, it is used as the primary photometric calibration.
      Gamma correction is then skipped (no double nonlinearity).
    - vignette_mask is assumed to be defined on the *unrectified* (distorted) image.
      Therefore, vignetting is corrected *before* undistortion.
    """

    def __init__(
        self,
        gamma: float = 2.2,
        crf_lut: np.ndarray | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        vignette_mask: np.ndarray | None = None,
        exposure_smooth_alpha: float = 0.1,
        denoise_method: str = "gaussian",  # "gaussian" (fast, default) or "bilateral"
    ):
        self.gamma = gamma
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.vignette_mask_raw = vignette_mask
        self.exposure_smooth_alpha = exposure_smooth_alpha
        self.denoise_method = denoise_method.lower()

        # ---- internal state ----
        self._ref_mean_luma = None          # for exposure compensation
        self._map1 = None                   # undistort map
        self._map2 = None
        self._vignette_mask = None          # resized vignette mask (HxW, float32)

        # precompute LUTs (0–255 → 0–255)
        self.gamma_lut = self._build_gamma_lut(gamma)
        self.crf_lut_u8 = self._build_crf_lut(crf_lut)

    # ---------- LUT builders ----------

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Build 8-bit gamma LUT so we don't do pow() every frame."""
        x = np.arange(256, dtype=np.float32) / 255.0
        x = np.clip(x, 1e-6, 1.0)
        y = np.power(x, gamma) * 255.0
        return np.clip(y, 0, 255).astype(np.uint8)

    @staticmethod
    def _build_crf_lut(crf_lut: np.ndarray | None) -> np.ndarray | None:
        """
        Convert user-provided CRF LUT to uint8 [0,255].
        Accepts 256-length array, either [0,1] float or [0,255] range.
        """
        if crf_lut is None:
            return None

        lut = np.asarray(crf_lut, dtype=np.float32)
        if lut.ndim != 1 or lut.shape[0] != 256:
            raise ValueError(f"crf_lut must be shape (256,), got {lut.shape}")

        if lut.max() <= 1.0 + 1e-6:
            lut = lut * 255.0

        return np.clip(lut, 0, 255).astype(np.uint8)

    # ---------- lazy init for maps / masks ----------

    def _ensure_undistort_map(self, width: int, height: int):
        if self._map1 is not None:
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        # use same camera_matrix as newCameraMatrix (no rectification)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            R=None,
            newCameraMatrix=self.camera_matrix,
            size=(width, height),
            m1type=cv2.CV_16SC2,
        )

    def _ensure_vignette_mask(self, width: int, height: int):
        if self.vignette_mask_raw is None:
            return
        if self._vignette_mask is not None:
            h, w = self._vignette_mask.shape[:2]
            if (w, h) == (width, height):
                return

        mask = self.vignette_mask_raw.astype(np.float32)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

        # ensure 2D single-channel
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # keep in (0,1], avoid zeros
        mask = np.clip(mask, 1e-3, 1.0)
        self._vignette_mask = mask

    # ---------- individual steps ----------

    def gamma_correction(self, img_u8: np.ndarray) -> np.ndarray:
        """
        LUT-based fixed gamma correction on uint8 BGR.

        If a CRF LUT is provided, we skip gamma correction to avoid
        stacking two nonlinear mappings.
        """
        if self.crf_lut_u8 is not None:
            # CRF handles photometric mapping; no gamma needed.
            return img_u8
        return cv2.LUT(img_u8, self.gamma_lut)

    def photometric_calibration(self, img_u8: np.ndarray) -> np.ndarray:
        """Optional CRF LUT; identity if no LUT."""
        if self.crf_lut_u8 is None:
            return img_u8
        return cv2.LUT(img_u8, self.crf_lut_u8)

    def exposure_compensation(self, img_u8: np.ndarray) -> np.ndarray:
        """
        Global exposure compensation based on mean luminance (EMA reference).
        Operates in float32 then converts back to uint8.
        """
        # use fast OpenCV gray (BT.601-ish) just for statistics
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())

        if self._ref_mean_luma is None:
            self._ref_mean_luma = mean_luma
            return img_u8

        # update reference mean (EMA)
        self._ref_mean_luma = (
            self.exposure_smooth_alpha * mean_luma
            + (1.0 - self.exposure_smooth_alpha) * self._ref_mean_luma
        )

        if mean_luma < 1e-3:
            return img_u8

        scale = self._ref_mean_luma / mean_luma

        img_f = img_u8.astype(np.float32) * scale
        img_f = np.clip(img_f, 0.0, 255.0)
        return img_f.astype(np.uint8)

    def remove_vignetting(self, img_u8: np.ndarray) -> np.ndarray:
        """
        Divide by precomputed vignetting mask (if exists).

        Assumes the mask is defined in the *distorted* image domain, so this
        should be called BEFORE undistortion.
        """
        if self._vignette_mask is None:
            return img_u8

        img_f = img_u8.astype(np.float32)
        mask = self._vignette_mask[:, :, None]  # (H,W,1) → broadcast to BGR
        img_f = img_f / mask
        img_f = np.clip(img_f, 0.0, 255.0)
        return img_f.astype(np.uint8)

    def undistort(self, img_u8: np.ndarray) -> np.ndarray:
        """Fast undistortion using precomputed remap."""
        if self._map1 is None or self._map2 is None:
            return img_u8
        return cv2.remap(img_u8, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def to_grayscale_bt709(img_u8: np.ndarray) -> np.ndarray:
        """
        Grayscale using BT.709 luminance:
        Y = 0.2126 R + 0.7152 G + 0.0722 B
        """
        b = img_u8[:, :, 0].astype(np.float32)
        g = img_u8[:, :, 1].astype(np.float32)
        r = img_u8[:, :, 2].astype(np.float32)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        y_u8 = np.clip(y, 0.0, 255.0).astype(np.uint8)
        return y_u8

    def light_denoise(self, gray_u8: np.ndarray) -> np.ndarray:
        """Fast denoising: Gaussian 3x3 (default), or optional bilateral."""
        if self.denoise_method == "bilateral":
            # slower, but you can enable it if you accept FPS loss
            return cv2.bilateralFilter(gray_u8, d=5, sigmaColor=20, sigmaSpace=5)
        else:
            # very fast, small kernel, keeps edges reasonably well
            return cv2.GaussianBlur(gray_u8, ksize=(3, 3), sigmaX=0)

    # ---------- full pipeline ----------

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Full fast baseline pipeline.

        Input
        -----
        frame_bgr : uint8 BGR

        Output
        ------
        gray_out : uint8, single-channel
        """
        assert frame_bgr.dtype == np.uint8, "input frame must be uint8 BGR"

        h, w = frame_bgr.shape[:2]
        self._ensure_undistort_map(w, h)
        self._ensure_vignette_mask(w, h)

        img = frame_bgr

        # 1. Photometric calibration:
        #    - If CRF LUT exists: CRF only (gamma skipped inside gamma_correction)
        #    - Else: use fixed gamma as a simple CRF approximation
        img = self.gamma_correction(img)
        img = self.photometric_calibration(img)

        # 2. Vignetting removal (mask defined on distorted image)
        img = self.remove_vignetting(img)

        # 3. Exposure compensation (global scale in linearized space)
        img = self.exposure_compensation(img)

        # 4. Undistortion (using precomputed maps)
        img = self.undistort(img)

        # 5. Grayscale (BT.709)
        gray = self.to_grayscale_bt709(img)

        # 6. Light denoising
        gray = self.light_denoise(gray)

        return gray

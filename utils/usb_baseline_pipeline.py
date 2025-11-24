# usb_baseline_pipeline.py
import cv2
import numpy as np


class USBBaselinePipeline:
    """
    USB Camera baseline pipeline:
    1. Gamma Correction (fixed gamma)
    2. Photometric Calibration (baseline: identity / optional LUT)
    3. Exposure Compensation (histogram/mean-based)
    4. Vignetting Removal (radial mask, optional)
    5. Undistortion (OpenCV camera matrix + distCoeffs)
    6. Grayscale Conversion (BT.709)
    7. Light Denoising (bilateral filter)
    """

    def __init__(
        self,
        gamma: float = 2.2,
        crf_lut: np.ndarray | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        vignette_mask: np.ndarray | None = None,
        exposure_smooth_alpha: float = 0.1,
    ):
        """
        Parameters
        ----------
        gamma : float
            Gamma for inverse gamma correction (baseline: 2.2 for sRGB).
        crf_lut : np.ndarray or None
            Optional 256-length LUT for photometric calibration (camera response function).
            If None, photometric_calibration() is identity.
        camera_matrix : np.ndarray or None
            3x3 camera intrinsic matrix for undistortion.
        dist_coeffs : np.ndarray or None
            Distortion coefficients (k1,k2,p1,p2[,k3,...]) for undistortion.
        vignette_mask : np.ndarray or None
            Precomputed vignetting mask (same HxW as frame, 1-channel float32, ~1.0 at center, smaller at corners).
        exposure_smooth_alpha : float
            EMA smoothing factor for reference mean exposure (0~1).
        """
        self.gamma = gamma
        self.crf_lut = crf_lut
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.vignette_mask = vignette_mask
        self.exposure_smooth_alpha = exposure_smooth_alpha

        # 內部 state：用來維持 exposure 的 reference mean
        self._ref_mean_luma = None

    # ========== 1. Gamma Correction ==========

    def gamma_correction(self, img: np.ndarray) -> np.ndarray:
        """
        Baseline: fixed inverse gamma correction.
        img: float32, range 0~1, BGR.
        """
        # 避免 0^gamma
        img = np.clip(img, 1e-6, 1.0)
        return np.power(img, self.gamma)

    # ========== 2. Photometric Calibration (baseline: identity / LUT) ==========

    def photometric_calibration(self, img: np.ndarray) -> np.ndarray:
        """
        Baseline: optional 1D LUT on [0,1] (per channel same curve).
        If crf_lut is None, this step is identity.

        img: float32, 0~1, BGR.
        """
        if self.crf_lut is None:
            return img

        # 假設 crf_lut.shape = (256,)，對 0~1 做 mapping
        lut = self.crf_lut.astype(np.float32)
        lut = np.clip(lut, 0.0, 1.0)

        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        # 對每個 channel 使用相同 LUT
        calibrated = cv2.LUT(img_uint8, lut)
        return calibrated.astype(np.float32) / 255.0

    # ========== 3. Exposure Compensation ==========

    def exposure_compensation(self, img: np.ndarray) -> np.ndarray:
        """
        Baseline: mean-based exposure normalization on luminance.
        img: float32, 0~1, BGR.
        """
        # 先轉 luminance 計算 mean (BT.709)
        luma = self.bt709_luminance(img)
        mean_luma = float(np.mean(luma))

        if self._ref_mean_luma is None:
            # 第一幀: 設 ref = 當前平均亮度
            self._ref_mean_luma = mean_luma
            return img

        # 更新 reference mean（平滑一下，避免抖動）
        self._ref_mean_luma = (
            self.exposure_smooth_alpha * mean_luma
            + (1.0 - self.exposure_smooth_alpha) * self._ref_mean_luma
        )

        if mean_luma < 1e-6:
            return img

        scale = self._ref_mean_luma / mean_luma
        img_scaled = img * scale
        img_scaled = np.clip(img_scaled, 0.0, 1.0)
        return img_scaled

    # ========== 4. Vignetting Removal ==========

    def remove_vignetting(self, img: np.ndarray) -> np.ndarray:
        """
        Baseline: divide by precomputed radial polynomial vignetting mask.
        img: float32, 0~1, BGR.
        vignette_mask: float32, 0~1, HxW (1-channel).
        """
        if self.vignette_mask is None:
            return img

        # 確保 shape match
        h, w = img.shape[:2]
        if self.vignette_mask.shape[:2] != (h, w):
            # 尺寸不符就先 resize
            mask = cv2.resize(self.vignette_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = self.vignette_mask

        # 避免除以 0
        mask = np.clip(mask, 1e-3, 1.0)
        # (H,W,1) broadcast 到 BGR
        if mask.ndim == 2:
            mask = mask[:, :, None]

        corrected = img / mask
        corrected = np.clip(corrected, 0.0, 1.0)
        return corrected

    # ========== 5. Undistortion ==========

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """
        Baseline: OpenCV undistort with camera_matrix & dist_coeffs.
        img: float32, 0~1, BGR.
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return img

        # OpenCV 需要 uint8 / float32 都可以，但會回傳同 dtype
        h, w = img.shape[:2]
        # 轉換到 0~255 uint8 再處理，比較常見
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        undistorted = cv2.undistort(img_u8, self.camera_matrix, self.dist_coeffs)
        undistorted = undistorted.astype(np.float32) / 255.0
        return undistorted

    # ========== 6. Grayscale Conversion (BT.709) ==========

    @staticmethod
    def bt709_luminance(img: np.ndarray) -> np.ndarray:
        """
        Compute luminance Y using BT.709 from BGR float32 0~1.
        Y = 0.2126 R + 0.7152 G + 0.0722 B
        """
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return y

    def to_grayscale_bt709(self, img: np.ndarray) -> np.ndarray:
        """
        Output uint8 grayscale image (0~255).
        img: float32, 0~1, BGR.
        """
        y = self.bt709_luminance(img)
        y_u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
        return y_u8

    # ========== 7. Light Denoising ==========

    @staticmethod
    def light_denoise(gray: np.ndarray) -> np.ndarray:
        """
        Baseline: bilateral filter with small kernel.
        gray: uint8 single-channel.
        """
        # (d, sigmaColor, sigmaSpace) 可再微調
        # d = 5 是一個常見的小 kernel
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=5)
        return denoised

    # ========== Pipeline wrapper ==========

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Full baseline pipeline.

        Input
        -----
        frame_bgr : np.ndarray
            uint8, BGR, 0~255.

        Output
        ------
        gray_out : np.ndarray
            uint8, single channel, 已經 gamma + photometric + exposure + vignetting + undistortion + gray + denoise
        """
        # 轉 float32 0~1
        img = frame_bgr.astype(np.float32) / 255.0

        # 1. Gamma Correction
        img = self.gamma_correction(img)

        # 2. Photometric Calibration (optional)
        img = self.photometric_calibration(img)

        # 3. Exposure Compensation
        img = self.exposure_compensation(img)

        # 4. Vignetting Removal
        img = self.remove_vignetting(img)

        # 5. Undistortion
        img = self.undistort(img)

        # 6. Grayscale Conversion (BT.709)
        gray = self.to_grayscale_bt709(img)

        # 7. Light Denoising
        gray = self.light_denoise(gray)

        return gray

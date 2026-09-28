"""Weather-adaptive image pre-processing for adverse driving conditions.

Supports fog/haze removal (Dark Channel Prior), rain streak suppression
(bilateral filter + morphological ops), and general contrast enhancement (CLAHE).
All techniques are pure OpenCV/NumPy — no extra model weights required.

Usage:
    from image_preprocessor import WeatherPreprocessor

    preprocessor = WeatherPreprocessor(mode="auto")
    clean_image = preprocessor.preprocess(bgr_image)
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

WeatherMode = Literal["fog", "haze", "rain", "clear", "auto"]


class WeatherPreprocessor:
    """Detects weather conditions and applies appropriate image restoration."""

    def __init__(self, mode: WeatherMode = "auto") -> None:
        self.mode: WeatherMode = mode
        LOGGER.info("WeatherPreprocessor initialised with mode=%s", self.mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Run weather-adaptive pre-processing on a BGR image.

        Returns:
            (processed_image, detected_weather_label)
        """
        if self.mode == "auto":
            weather = self._detect_weather(image)
        else:
            weather = self.mode

        if weather in ("fog", "haze"):
            image = self._dehaze_dcp(image)
        elif weather == "rain":
            image = self._derain(image)
        # "clear" → skip heavy restoration

        # Always apply a light CLAHE polish for low-visibility conditions
        if weather != "clear":
            image = self._apply_clahe(image)

        return image, weather

    # ------------------------------------------------------------------
    # Weather auto-detection from image statistics
    # ------------------------------------------------------------------

    def _detect_weather(self, image: np.ndarray) -> str:
        """Estimate weather from simple image statistics.

        Heuristics used:
        - **Fog / haze**: high mean intensity + low saturation variance
          (the scene looks uniformly washed out).
        - **Rain**: moderate intensity + high edge density in vertical
          direction (rain streaks create strong vertical gradients).
        - **Clear**: everything else.

        These thresholds work well for CARLA renders; you may need to
        tune them for real-world cameras.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mean_v = float(np.mean(v))
        sat_var = float(np.var(s))

        # Vertical Sobel to detect rain-streak-like gradients
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        vert_edge_density = float(np.mean(np.abs(sobel_v)))

        # --- Decision logic ---
        # Fog / haze: bright but low-saturation
        if mean_v > 120 and sat_var < 1800:
            return "fog"

        # Rain: moderate brightness with strong vertical edges
        if vert_edge_density > 18 and sat_var < 3500:
            return "rain"

        return "clear"

    # ------------------------------------------------------------------
    # Dark Channel Prior dehazing  (He et al., 2009)
    # ------------------------------------------------------------------

    @staticmethod
    def _dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
        """Compute the dark channel of a BGR image."""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (patch_size, patch_size)
        )
        dark = cv2.erode(min_channel, kernel)
        return dark

    @staticmethod
    def _estimate_atmospheric_light(
        image: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001
    ) -> np.ndarray:
        """Pick the atmospheric light from the brightest pixels in the dark channel."""
        num_pixels = dark_channel.size
        num_top = max(int(num_pixels * top_percent), 1)
        flat_dark = dark_channel.ravel()
        indices = np.argpartition(flat_dark, -num_top)[-num_top:]
        flat_image = image.reshape(-1, 3)
        brightest = flat_image[indices]
        atm_light = np.max(brightest, axis=0).astype(np.float64)
        return atm_light

    def _dehaze_dcp(
        self,
        image: np.ndarray,
        omega: float = 0.75,
        t_min: float = 0.1,
        patch_size: int = 15,
        guided_radius: int = 60,
        guided_eps: float = 1e-3,
    ) -> np.ndarray:
        """Remove fog / haze using Dark Channel Prior.

        Args:
            omega:  how much haze to remove (0-1). 0.75 keeps a slight
                    atmospheric perspective which looks more natural.
            t_min:  minimum transmission to prevent over-amplification.
        """
        img_f = image.astype(np.float64)
        dark = self._dark_channel(image, patch_size)
        atm = self._estimate_atmospheric_light(image, dark)

        # Normalise image by atmospheric light and compute transmission
        normalised = img_f / (atm + 1e-6)
        dark_norm = self._dark_channel(
            normalised.astype(np.float64).clip(0, 1) * 255, patch_size
        ) / 255.0
        transmission = 1.0 - omega * dark_norm
        transmission = np.clip(transmission, t_min, 1.0)

        # Guided filter to refine the transmission map (edge-aware smoothing)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        transmission = self._guided_filter(
            gray, transmission, guided_radius, guided_eps
        )
        transmission = np.clip(transmission, t_min, 1.0)

        # Recover scene radiance
        t_map = transmission[:, :, np.newaxis]
        result = (img_f - atm) / t_map + atm
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _guided_filter(
        guide: np.ndarray,
        src: np.ndarray,
        radius: int,
        eps: float,
    ) -> np.ndarray:
        """Edge-preserving guided filter (O(N) box-filter implementation)."""
        ksize = (2 * radius + 1, 2 * radius + 1)
        mean_g = cv2.boxFilter(guide, cv2.CV_64F, ksize)
        mean_s = cv2.boxFilter(src, cv2.CV_64F, ksize)
        corr_gs = cv2.boxFilter(guide * src, cv2.CV_64F, ksize)
        var_g = cv2.boxFilter(guide * guide, cv2.CV_64F, ksize) - mean_g * mean_g

        a = (corr_gs - mean_g * mean_s) / (var_g + eps)
        b = mean_s - a * mean_g
        mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)
        return mean_a * guide + mean_b

    # ------------------------------------------------------------------
    # Rain streak removal
    # ------------------------------------------------------------------

    @staticmethod
    def _derain(image: np.ndarray) -> np.ndarray:
        """Suppress rain streaks using bilateral filtering + morphological ops.

        Rain streaks are thin, high-frequency, roughly vertical structures.
        The bilateral filter smooths them while preserving strong object edges,
        and a morphological close fills any residual thin gaps.
        """
        # Bilateral filter — large colour sigma to blur streak colours together
        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Morphological close with a small vertical kernel to fill thin gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # Blend: keep 80% filtered + 20% original to preserve texture
        result = cv2.addWeighted(closed, 0.80, image, 0.20, 0)
        return result

    # ------------------------------------------------------------------
    # CLAHE contrast enhancement
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_clahe(
        image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8
    ) -> np.ndarray:
        """Apply CLAHE on the L channel of LAB colour space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
        )
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


__all__ = ["WeatherPreprocessor", "WeatherMode"]

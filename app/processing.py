from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """Apply gray-world white balance to reduce color cast from uneven lighting."""
    img = image.astype(np.float32)
    means = img.mean(axis=(0, 1))
    gray = float(np.mean(means)) + 1e-6
    scales = gray / (means + 1e-6)
    balanced = img * scales
    return np.clip(balanced, 0, 255).astype(np.uint8)


@lru_cache(maxsize=16)
def _get_clahe(clip_limit: float, tile_grid: int) -> cv2.CLAHE:
    clip = max(0.1, float(clip_limit))
    grid = max(2, int(tile_grid))
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))


def normalize_illumination(
    image: np.ndarray,
    *,
    clip_limit: float = 2.4,
    tile_grid: int = 8,
    apply_gray_world: bool = True,
    gray_world_strength: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Balance color and locally equalize illumination.

    Returns: (normalized_bgr, equalized_l_channel, white_balanced_bgr).
    """
    balanced = image
    strength = float(np.clip(gray_world_strength, 0.0, 1.0))
    if apply_gray_world and strength > 0.0:
        wb = gray_world_white_balance(image)
        if strength >= 0.999:
            balanced = wb
        else:
            balanced = cv2.addWeighted(wb, strength, image, 1.0 - strength, 0.0)

    clahe = _get_clahe(round(float(clip_limit), 2), int(tile_grid))
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    l_equalized = clahe.apply(l_channel)
    lab[:, :, 0] = l_equalized

    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return normalized, l_equalized, balanced


def suppress_red_text(img_bgr: np.ndarray) -> np.ndarray:
    """Return mask of printed red text/graphics to ignore in counting."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (np.array([0, 60, 60], np.uint8), np.array([12, 255, 255], np.uint8))
    lower2, upper2 = (np.array([168, 60, 60], np.uint8), np.array([180, 255, 255], np.uint8))
    red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)
    return red


def locate_yellow_pad(
    image_bgr: np.ndarray,
    *,
    margin_px: int = 6,
    min_area_frac: float = 0.12,
) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detect the primary yellow gluepad region using color segmentation.
    Returns ((x0, y0, x1, y1), mask) or (None, mask) if not found.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 70, 80], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    h, w = image_bgr.shape[:2]
    min_area = float(max(1.0, min_area_frac * h * w))
    best_rect = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area <= best_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        best_rect = (x, y, x + width, y + height)
        best_area = area

    if best_rect is None:
        return None, mask

    margin = max(0, int(margin_px))
    x0, y0, x1, y1 = best_rect
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    return (x0, y0, x1, y1), mask


def auto_crop_noise_bands(peaks: np.ndarray, max_crop_frac: float = 0.35, mult: float = 2.2) -> Tuple[int, int]:
    """Crop dense horizontal noise bands. Returns (y_start, y_end) indices to keep."""
    H, W = peaks.shape[:2]
    row_density = peaks.sum(axis=1) / 255.0
    m0, m1 = int(0.40 * H), int(0.60 * H)
    baseline = np.median(row_density[m0:m1]) + 1e-6
    k = max(3, H // 200)
    smooth = np.convolve(row_density, np.ones(k) / k, mode="same")

    top_limit, bot_limit = int(H * max_crop_frac), H - int(H * max_crop_frac)
    y_top = 0
    while y_top < top_limit and smooth[y_top] > mult * baseline:
        y_top += 1
    y_bot = H - 1
    while y_bot > bot_limit and smooth[y_bot] > mult * baseline:
        y_bot -= 1

    y_start, y_end = y_top, y_bot + 1
    if y_end - y_start < int(0.5 * H):
        y_start, y_end = int(0.1 * H), int(0.9 * H)
    return y_start, y_end


def find_title_rule_y(
    img_bgr: np.ndarray,
    *,
    top_frac: float = 0.6,
    min_len_frac: float = 0.55,
    angle_tol_deg: float = 6.0,
) -> Optional[int]:
    """
    Detect the long horizontal rule under the title using HoughLinesP.
    Returns y (int) of the line if found, else None.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=int(min_len_frac * w),
        maxLineGap=10,
    )
    if lines is None:
        return None

    best_y = None
    best_len = 0
    top_limit = int(top_frac * h)
    for x1, y1, x2, y2 in lines[:, 0]:
        dy, dx = abs(y2 - y1), abs(x2 - x1)
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > angle_tol_deg or max(y1, y2) > top_limit:
            continue
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length > best_len:
            best_len = length
            best_y = int((y1 + y2) / 2)
    return best_y


def _integral_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def detect_dark_blobs(
    image: np.ndarray,
    *,
    annotate: bool = False,
    bg_kernel: int = 41,
    peak_thresh: float = 0.36,
    min_support_area: int = 6,
    min_blob_area: int = 1,
    use_pad_crop: bool = True,
    pad_crop_margin_px: int = 6,
    use_red_suppression: bool = True,
    use_title_line_crop: bool = True,
    title_line_margin_px: int = 12,
    auto_crop: bool = True,
    max_crop_frac: float = 0.35,
    crop_density_mult: float = 2.2,
    crop_top_pct: float = 0.0,
    crop_bottom_pct: float = 0.0,
    enable_gray_world: bool = True,
    gray_world_strength: float = 1.0,
    clahe_clip_limit: float = 2.4,
    clahe_tile_grid: int = 8,
    area_pad_px: int = 2,
) -> Tuple[int, Optional[np.ndarray], Dict[str, np.ndarray | int]]:
    """
    Robust counter for yellow gluepads. Returns (count, annotated, debug).
    Applies white balance and CLAHE to stabilize illumination before detection.
    """
    min_blob_area = max(1, int(min_blob_area))
    min_support_area = max(0, int(min_support_area))
    orig_image = image
    processed_bgr, L_eq, balanced = normalize_illumination(
        image,
        clip_limit=clahe_clip_limit,
        tile_grid=clahe_tile_grid,
        apply_gray_world=enable_gray_world,
        gray_world_strength=gray_world_strength,
    )

    H0, W0 = processed_bgr.shape[:2]
    y_pre_start = 0
    pad_bounds: Optional[Tuple[int, int, int, int]] = None
    pad_mask = np.zeros((H0, W0), dtype=np.uint8)
    if use_pad_crop:
        pad_bounds, pad_mask = locate_yellow_pad(
            processed_bgr,
            margin_px=int(pad_crop_margin_px),
        )

    if use_title_line_crop:
        y_rule = find_title_rule_y(processed_bgr)
        if y_rule is not None:
            y_pre_start = max(0, y_rule + int(title_line_margin_px))

    k = bg_kernel if bg_kernel % 2 == 1 else bg_kernel + 1
    bg = cv2.medianBlur(L_eq, k)
    enhanced = cv2.subtract(bg, L_eq)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    _, thr = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    red_mask = suppress_red_text(processed_bgr) if use_red_suppression else np.zeros_like(thr)
    thr = cv2.bitwise_and(thr, cv2.bitwise_not(red_mask))

    dist = cv2.distanceTransform(thr, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, peaks = cv2.threshold(dist_norm, float(peak_thresh), 1.0, cv2.THRESH_BINARY)
    peaks = (peaks * 255).astype(np.uint8)

    H, W = peaks.shape[:2]
    x_start, x_end = 0, W
    y_start, y_end = int(y_pre_start), H

    if pad_bounds is not None:
        px0, py0, px1, py1 = pad_bounds
        x_start = max(x_start, int(px0))
        x_end = min(x_end, int(px1))
        y_start = max(y_start, int(py0))
        y_end = min(y_end, int(py1))
        if x_end - x_start < 5 or y_end - y_start < 5:
            x_start, x_end = 0, W
            y_start, y_end = int(y_pre_start), H

    if auto_crop and y_start < y_end and x_start < x_end:
        cropped = peaks[y_start:y_end, x_start:x_end]
        if cropped.size > 0:
            ys, ye = auto_crop_noise_bands(cropped, max_crop_frac=max_crop_frac, mult=crop_density_mult)
            y_start += ys
            y_end = y_start + (ye - ys)

    if crop_top_pct > 0:
        y_start = max(y_start, int(H * crop_top_pct))
    if crop_bottom_pct > 0:
        y_end = min(y_end, H - int(H * crop_bottom_pct))

    y_start = max(0, min(y_start, H))
    y_end = max(y_start + 1, min(y_end, H))
    x_start = max(0, min(x_start, W - 1))
    x_end = max(x_start + 1, min(x_end, W))

    thr_c = thr[y_start:y_end, x_start:x_end]
    peaks_c = peaks[y_start:y_end, x_start:x_end]

    peaks_binary = (peaks_c > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks_binary, connectivity=8)

    thr_bool = (thr_c > 0).astype(np.uint8)
    thr_integral = cv2.integral(thr_bool, sdepth=cv2.CV_32S)
    pad = max(0, int(area_pad_px))

    count = 0
    annotated = orig_image.copy() if annotate else None

    height_c, width_c = thr_c.shape[:2]
    for label in range(1, num_labels):
        x, y, w, h, comp_area = stats[label]
        comp_area = int(comp_area)
        if comp_area < min_blob_area:
            continue

        x0 = max(int(x) - pad, 0)
        y0 = max(int(y) - pad, 0)
        x1 = min(int(x + w + pad), width_c)
        y1 = min(int(y + h + pad), height_c)

        area = _integral_sum(thr_integral, x0, y0, x1, y1)
        if area < min_support_area:
            continue

        count += 1

        if annotate and annotated is not None:
            cx, cy = centroids[label]
            cx_i = int(round(cx)) + x_start
            cy_i = int(round(cy)) + y_start
            if 0 <= cx_i < W0 and 0 <= cy_i < H0:
                cv2.circle(annotated, (cx_i, cy_i), 8, (0, 0, 255), 1)
                cv2.putText(
                    annotated,
                    str(count),
                    (cx_i + 3, cy_i - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                )

    debug: Dict[str, np.ndarray | int] = {
        "balanced": balanced,
        "normalized": processed_bgr,
        "L": L_eq,
        "bg": bg,
        "enhanced": enhanced,
        "thr": thr,
        "dist_norm": (dist_norm * 255).astype(np.uint8),
        "peaks": peaks,
        "red_mask": red_mask,
        "pre_rule_y": int(y_pre_start),
        "y_start": int(y_start),
        "y_end": int(y_end),
        "x_start": int(x_start),
        "x_end": int(x_end),
        "min_blob_area": int(min_blob_area),
        "pad_mask": pad_mask,
    }
    return count, annotated, debug


__all__ = [
    "detect_dark_blobs",
    "normalize_illumination",
    "gray_world_white_balance",
]

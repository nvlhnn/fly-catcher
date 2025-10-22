from __future__ import annotations
from typing import Tuple, Optional
import cv2
import numpy as np


def detect_dark_blobs(
    image: np.ndarray,
    *,
    annotate: bool = False,
    bg_kernel: int = 31,          # odd, large (31–61) to remove illumination
    peak_thresh: float = 0.35,    # 0.30–0.45; lower = more peaks (splits clusters)
    min_support_area: int = 6,    # discard tiny specks
) -> Tuple[int, Optional[np.ndarray], dict]:
    """
    Count dark insects on yellow gluepads, robust to glare & clusters.
    Returns: (count, annotated_bgr_or_None, debug_images)
    debug_images -> dict with intermediate single-channel images for /debug.
    """

    # ---- 1) Work in LAB (L channel ~ brightness) ----
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # ---- 2) Background removal (black-hat via bg - L) ----
    k = bg_kernel if bg_kernel % 2 == 1 else bg_kernel + 1
    bg = cv2.medianBlur(L, k)
    enhanced = cv2.subtract(bg, L)  # darker-than-bg -> bright

    # Normalize for stability
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # ---- 3) Otsu threshold on enhanced map ----
    _, thr = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Small cleanups
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ---- 4) Distance transform -> get one peak per insect ----
    dist = cv2.distanceTransform(thr, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, peaks = cv2.threshold(dist_norm, float(peak_thresh), 1.0, cv2.THRESH_BINARY)
    peaks = (peaks * 255).astype(np.uint8)

    # ---- 5) Count connected components of peaks ----
    num_labels, labels = cv2.connectedComponents(peaks)

    count = 0
    annotated = image.copy() if annotate else None

    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8) * 255

        # support area in thresholded mask (ignore micro noise)
        support = cv2.bitwise_and(thr, thr, mask=mask)
        area = cv2.countNonZero(support)
        if area < min_support_area:
            continue

        count += 1

        if annotate and annotated is not None:
            M = cv2.moments(mask)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), 1)
            cv2.putText(annotated, str(count), (cx + 3, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    debug = {
        "L": L,
        "bg": bg,
        "enhanced": enhanced,
        "thr": thr,
        "dist_norm": (dist_norm * 255).astype(np.uint8),
        "peaks": peaks,
    }
    return count, annotated, debug

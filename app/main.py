# main.py
from __future__ import annotations

import base64, time
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Gluepad Fly Counter", version="1.4.0")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                  # tighten for prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],    # allow browsers to read it
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- Helpers ----
MAX_PIXELS = 4_000_000  # ~4MP cap

async def read_image_or_400(file: UploadFile) -> np.ndarray:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Image file is empty")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unable to decode the image")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        s = (MAX_PIXELS / (h * w)) ** 0.5
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img

def suppress_red_text(img_bgr: np.ndarray) -> np.ndarray:
    """Return mask of red text/graphics."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (np.array([0, 60, 60], np.uint8),  np.array([12, 255, 255], np.uint8))
    lower2, upper2 = (np.array([168, 60, 60], np.uint8), np.array([180, 255, 255], np.uint8))
    red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)
    return red

def auto_crop_noise_bands(peaks: np.ndarray, max_crop_frac: float = 0.35, mult: float = 2.2) -> tuple[int, int]:
    """Crop dense horizontal bands (fallback). Returns [y_start, y_end] to keep."""
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
    if y_end - y_start < int(0.5 * H):  # safety
        y_start, y_end = int(0.1 * H), int(0.9 * H)
    return y_start, y_end

def find_title_rule_y(img_bgr: np.ndarray,
                      top_frac: float = 0.6,
                      min_len_frac: float = 0.55,
                      angle_tol_deg: float = 6.0) -> Optional[int]:
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
        if dx == 0:  # vertical
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > angle_tol_deg:  # not horizontal enough
            continue
        if max(y1, y2) > top_limit:  # we only care about the top half-ish
            continue
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length > best_len:
            best_len = length
            best_y = int((y1 + y2) / 2)
    return best_y

# ---- CV core ----
def detect_dark_blobs(
    image: np.ndarray,
    *,
    annotate: bool = False,
    bg_kernel: int = 41,             # odd 31–61
    peak_thresh: float = 0.36,
    min_support_area: int = 6,
    use_red_suppression: bool = True,
    use_title_line_crop: bool = True,       # NEW: crop above detected rule
    title_line_margin_px: int = 12,         # extra pixels below rule to crop
    auto_crop: bool = True,
    max_crop_frac: float = 0.35,
    crop_density_mult: float = 2.2,
    crop_top_pct: float = 0.0,
    crop_bottom_pct: float = 0.0,
) -> Tuple[int, Optional[np.ndarray], Dict[str, np.ndarray | int]]:
    """
    Robust counter for yellow gluepads. Returns (count, annotated, debug).
    """

    # 0) Optional: detect the horizontal rule and pre-crop
    H0, W0 = image.shape[:2]
    y_pre_start = 0
    if use_title_line_crop:
        y_rule = find_title_rule_y(image)
        if y_rule is not None:
            y_pre_start = max(0, y_rule + int(title_line_margin_px))

    # 1) LAB black-hat enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    k = bg_kernel if bg_kernel % 2 == 1 else bg_kernel + 1
    bg = cv2.medianBlur(L, k)
    enhanced = cv2.subtract(bg, L)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # 2) Threshold + clean
    _, thr = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) Suppress printed red text
    red_mask = suppress_red_text(image) if use_red_suppression else np.zeros_like(thr)
    thr = cv2.bitwise_and(thr, cv2.bitwise_not(red_mask))

    # 4) Distance transform peaks
    dist = cv2.distanceTransform(thr, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, peaks = cv2.threshold(dist_norm, float(peak_thresh), 1.0, cv2.THRESH_BINARY)
    peaks = (peaks * 255).astype(np.uint8)

    # 5) Crop region to count
    H, W = peaks.shape[:2]
    y_start, y_end = int(y_pre_start), H  # ensure rule crop is applied
    if auto_crop:
        # auto-crop within the already rule-cropped region
        pc = peaks[y_start:y_end, :]
        ys, ye = auto_crop_noise_bands(pc, max_crop_frac=max_crop_frac, mult=crop_density_mult)
        y_start += ys
        y_end = y_start + (ye - ys)

    if crop_top_pct > 0:
        y_start = max(y_start, int(H * crop_top_pct))
    if crop_bottom_pct > 0:
        y_end = min(y_end, H - int(H * crop_bottom_pct))

    # 6) Count connected components on cropped peaks
    thr_c = thr[y_start:y_end, :]
    peaks_c = peaks[y_start:y_end, :]
    num_labels, labels = cv2.connectedComponents(peaks_c)

    count = 0
    annotated = image.copy() if annotate else None

    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8) * 255
        support = cv2.bitwise_and(thr_c, thr_c, mask=mask)
        area = cv2.countNonZero(support)
        if area < min_support_area:
            continue
        count += 1

        if annotate and annotated is not None:
            M = cv2.moments(mask)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + y_start
            cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), 1)
            cv2.putText(annotated, str(count), (cx + 3, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    debug = {
        "L": L, "bg": bg, "enhanced": enhanced, "thr": thr,
        "dist_norm": (dist_norm * 255).astype(np.uint8), "peaks": peaks,
        "red_mask": red_mask,
        "pre_rule_y": int(y_pre_start),
        "y_start": int(y_start), "y_end": int(y_end),
    }
    return count, annotated, debug

# ---- Endpoints ----
@app.post("/count")
async def count_endpoint(
    file: UploadFile = File(...),
    return_marked: bool = Form(False),
    bg_kernel: int = Form(41),
    peak_thresh: float = Form(0.36),
    min_support_area: int = Form(6),
    use_red_suppression: bool = Form(True),
    use_title_line_crop: bool = Form(True),
    title_line_margin_px: int = Form(12),
    auto_crop: bool = Form(True),
    max_crop_frac: float = Form(0.35),
    crop_density_mult: float = Form(2.2),
    crop_top_pct: float = Form(0.0),
    crop_bottom_pct: float = Form(0.0),
):
    t0 = time.time()
    img = await read_image_or_400(file)
    count, marked, _ = detect_dark_blobs(
        img,
        annotate=return_marked,
        bg_kernel=bg_kernel,
        peak_thresh=peak_thresh,
        min_support_area=min_support_area,
        use_red_suppression=use_red_suppression,
        use_title_line_crop=use_title_line_crop,
        title_line_margin_px=title_line_margin_px,
        auto_crop=auto_crop,
        max_crop_frac=max_crop_frac,
        crop_density_mult=crop_density_mult,
        crop_top_pct=crop_top_pct,
        crop_bottom_pct=crop_bottom_pct,
    )
    print(f"[/count] {count} in {time.time()-t0:.3f}s")

    headers = {"X-Total-Count": str(count)}
    if return_marked:
        ok, buf = cv2.imencode(".png", marked)
        if not ok:
            raise HTTPException(status_code=500, detail="PNG encoding failed")
        headers["Content-Disposition"] = "inline; filename=marked.png"
        return Response(content=buf.tobytes(), media_type="image/png", headers=headers)

    return JSONResponse(content={"count": int(count)}, headers=headers)

@app.post("/debug")
async def debug_endpoint(
    file: UploadFile = File(...),
    bg_kernel: int = Form(41),
    peak_thresh: float = Form(0.36),
    min_support_area: int = Form(6),
    use_red_suppression: bool = Form(True),
    use_title_line_crop: bool = Form(True),
    title_line_margin_px: int = Form(12),
    auto_crop: bool = Form(True),
    max_crop_frac: float = Form(0.35),
    crop_density_mult: float = Form(2.2),
    crop_top_pct: float = Form(0.0),
    crop_bottom_pct: float = Form(0.0),
):
    t0 = time.time()
    img = await read_image_or_400(file)
    count, marked, dbg = detect_dark_blobs(
        img, annotate=True,
        bg_kernel=bg_kernel, peak_thresh=peak_thresh, min_support_area=min_support_area,
        use_red_suppression=use_red_suppression,
        use_title_line_crop=use_title_line_crop, title_line_margin_px=title_line_margin_px,
        auto_crop=auto_crop, max_crop_frac=max_crop_frac, crop_density_mult=crop_density_mult,
        crop_top_pct=crop_top_pct, crop_bottom_pct=crop_bottom_pct,
    )
    print(f"[/debug] {count} in {time.time()-t0:.3f}s")

    def b64png(arr: np.ndarray) -> str:
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", arr)
        return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8") if ok else ""

    payload = {
        "count": int(count),
        "preview": {
            "marked": b64png(marked),
            "L": b64png(dbg["L"]), "bg": b64png(dbg["bg"]),
            "enhanced": b64png(dbg["enhanced"]), "thr": b64png(dbg["thr"]),
            "dist_norm": b64png(dbg["dist_norm"]), "peaks": b64png(dbg["peaks"]),
            "red_mask": b64png(dbg["red_mask"]),
            "pre_rule_y": dbg["pre_rule_y"],
            "auto_crop_y": [dbg["y_start"], dbg["y_end"]],
        },
        "params": {
            "bg_kernel": bg_kernel, "peak_thresh": peak_thresh, "min_support_area": min_support_area,
            "use_red_suppression": use_red_suppression,
            "use_title_line_crop": use_title_line_crop, "title_line_margin_px": title_line_margin_px,
            "auto_crop": auto_crop, "max_crop_frac": max_crop_frac, "crop_density_mult": crop_density_mult,
            "crop_top_pct": crop_top_pct, "crop_bottom_pct": crop_bottom_pct,
        },
    }
    return JSONResponse(content=payload, headers={"X-Total-Count": str(count)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

# main.py
from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .processing import detect_dark_blobs

app = FastAPI(title="Gluepad Fly Counter", version="1.5.0")

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
    enable_gray_world: bool = Form(True),
    gray_world_strength: float = Form(1.0),
    clahe_clip_limit: float = Form(2.4),
    clahe_tile_grid: int = Form(8),
    area_pad_px: int = Form(2),
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
        enable_gray_world=enable_gray_world,
        gray_world_strength=gray_world_strength,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
        area_pad_px=area_pad_px,
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
    enable_gray_world: bool = Form(True),
    gray_world_strength: float = Form(1.0),
    clahe_clip_limit: float = Form(2.4),
    clahe_tile_grid: int = Form(8),
    area_pad_px: int = Form(2),
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
        enable_gray_world=enable_gray_world, gray_world_strength=gray_world_strength,
        clahe_clip_limit=clahe_clip_limit, clahe_tile_grid=clahe_tile_grid,
        area_pad_px=area_pad_px,
    )
    print(f"[/debug] {count} in {time.time()-t0:.3f}s")

    def b64png(arr) -> str:
        if arr is None:
            return ""
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", arr)
        return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8") if ok else ""

    payload = {
        "count": int(count),
        "preview": {
            "marked": b64png(marked),
            "balanced": b64png(dbg["balanced"]),
            "normalized": b64png(dbg["normalized"]),
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
            "enable_gray_world": enable_gray_world,
            "gray_world_strength": gray_world_strength,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid": clahe_tile_grid,
            "area_pad_px": area_pad_px,
        },
    }
    return JSONResponse(content=payload, headers={"X-Total-Count": str(count)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

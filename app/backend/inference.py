import asyncio
import io
import logging
import math
import os
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import tensorflow as tf

load_dotenv()

logger = logging.getLogger(__name__)

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "track_model.keras"

# Zoom 16 at 512px gives ~1.2km coverage per tile, matching the ~1km NAIP training chips
ZOOM = 16
TILE_PX = 512
IMG_SIZE = (224, 224)
GRID_SPACING_KM = 1.0
CONCURRENCY = 8

_model = None


def load_model() -> tf.keras.Model:
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Model loaded.")
    return _model


def _grid_points(
    center_lat: float, center_lng: float, radius_km: float
) -> list[tuple[float, float]]:
    R = 6371.0
    points = []
    n = math.ceil(radius_km / GRID_SPACING_KM)
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            dist = math.sqrt(
                (di * GRID_SPACING_KM) ** 2 + (dj * GRID_SPACING_KM) ** 2
            )
            if dist > radius_km:
                continue
            dlat = (di * GRID_SPACING_KM / R) * (180.0 / math.pi)
            dlng = (
                (dj * GRID_SPACING_KM / R)
                * (180.0 / math.pi)
                / math.cos(math.radians(center_lat))
            )
            points.append((center_lat + dlat, center_lng + dlng))
    return points


async def _fetch_tile(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    lat: float,
    lng: float,
) -> np.ndarray | None:
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lng:.6f},{lat:.6f},{ZOOM}/{TILE_PX}x{TILE_PX}"
        f"?access_token={MAPBOX_TOKEN}"
    )
    async with sem:
        try:
            resp = await client.get(url, timeout=15.0)
            if resp.status_code != 200:
                logger.warning(
                    "Mapbox tile fetch failed: HTTP %d | lat=%.4f lng=%.4f | body: %s",
                    resp.status_code,
                    lat,
                    lng,
                    resp.text[:200],
                )
                return None
            img = (
                Image.open(io.BytesIO(resp.content))
                .convert("RGB")
                .resize(IMG_SIZE, Image.Resampling.LANCZOS)
            )
            return np.array(img, dtype=np.float32)
        except Exception as exc:
            logger.warning("Tile fetch exception lat=%.4f lng=%.4f: %s", lat, lng, exc)
            return None


async def scan_area(
    lat: float,
    lng: float,
    radius_km: float,
    threshold: float = 0.65,
) -> dict:
    points = _grid_points(lat, lng, radius_km)
    model = load_model()
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient() as client:
        tasks = [_fetch_tile(client, sem, p[0], p[1]) for p in points]
        images = await asyncio.gather(*tasks)

    valid_points: list[tuple[float, float]] = []
    valid_imgs: list[np.ndarray] = []
    for pt, img in zip(points, images):
        if img is not None:
            valid_points.append(pt)
            valid_imgs.append(img)

    if not valid_imgs:
        return {"detections": [], "tiles_scanned": 0}

    batch = np.stack(valid_imgs)
    preds = model.predict(batch, batch_size=16, verbose=0).flatten()

    detections = []
    for (plat, plng), conf in zip(valid_points, preds):
        if conf >= threshold:
            detections.append(
                {"lat": plat, "lng": plng, "confidence": round(float(conf), 3)}
            )

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return {"detections": detections, "tiles_scanned": len(valid_imgs)}

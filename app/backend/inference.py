import asyncio
import io
import logging
import math
import os
from pathlib import Path

import asyncpg
import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import tensorflow as tf

load_dotenv()

logger = logging.getLogger(__name__)

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")

_model_path_env = os.getenv("MODEL_PATH")
MODEL_PATH = (
    Path(_model_path_env)
    if _model_path_env
    else Path(__file__).resolve().parent / "track_model.keras"
)

ZOOM = 16
TILE_PX = 512
IMG_SIZE = (224, 224)
GRID_SPACING_KM = 0.5
CONCURRENCY = 16

NEIGHBOR_RADIUS_KM = 1.5
PRELIM_SCALE      = 0.7
OWN_WEIGHT        = 0.5
NEIGHBOR_WEIGHT   = 0.5
NEIGHBOR_FLOOR    = 0.3
CLUSTER_RADIUS_KM = 1.2

_model = None


def load_model() -> tf.keras.Model:
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Model loaded.")
    return _model


def _lat_lng_to_tile(lat: float, lng: float, zoom: int = ZOOM) -> tuple[int, int, int]:
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return zoom, x, y


def _tile_center(z: int, x: int, y: int) -> tuple[float, float]:
    n = 2 ** z
    lng = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    return math.degrees(lat_rad), lng


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


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _build_neighbor_map(
    points: list[tuple[float, float]], radius_km: float = NEIGHBOR_RADIUS_KM
) -> dict[int, list[int]]:
    neighbor_map: dict[int, list[int]] = {i: [] for i in range(len(points))}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _haversine(points[i][0], points[i][1], points[j][0], points[j][1]) <= radius_km:
                neighbor_map[i].append(j)
                neighbor_map[j].append(i)
    return neighbor_map


def _aggregate_scores(
    scores: np.ndarray,
    neighbor_map: dict[int, list[int]],
    prelim_threshold: float,
) -> np.ndarray:
    agg = scores.copy()
    for i, score in enumerate(scores):
        if score < prelim_threshold:
            continue
        neighbor_scores = [scores[j] for j in neighbor_map[i] if scores[j] >= NEIGHBOR_FLOOR]
        neighbor_mean = sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0
        blended = OWN_WEIGHT * score + NEIGHBOR_WEIGHT * neighbor_mean
        agg[i] = max(score, blended)
    return agg


def _cluster_detections(
    candidates: list[tuple[float, float, float]], cluster_radius_km: float = CLUSTER_RADIUS_KM
) -> list[dict]:
    candidates = sorted(candidates, key=lambda c: c[2], reverse=True)
    consumed = [False] * len(candidates)
    detections = []
    for i, (ilat, ilng, iscore) in enumerate(candidates):
        if consumed[i]:
            continue
        cluster = [(ilat, ilng, iscore)]
        consumed[i] = True
        for j, (jlat, jlng, jscore) in enumerate(candidates):
            if consumed[j]:
                continue
            if _haversine(ilat, ilng, jlat, jlng) <= cluster_radius_km:
                cluster.append((jlat, jlng, jscore))
                consumed[j] = True
        total_weight = sum(c[2] for c in cluster)
        centroid_lat = sum(c[0] * c[2] for c in cluster) / total_weight
        centroid_lng = sum(c[1] * c[2] for c in cluster) / total_weight
        detections.append({
            "lat": centroid_lat,
            "lng": centroid_lng,
            "confidence": round(max(c[2] for c in cluster), 3),
        })
    return detections


async def _fetch_tile(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    lat: float,
    lng: float,
) -> np.ndarray | None:
    """Returns a 224×224 image array or None on failure."""
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
                    resp.status_code, lat, lng, resp.text[:200],
                )
                return None
            img_orig = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return np.array(
                img_orig.resize(IMG_SIZE, Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
        except Exception as exc:
            logger.warning("Tile fetch exception lat=%.4f lng=%.4f: %s", lat, lng, exc)
            return None


async def scan_area(
    lat: float,
    lng: float,
    radius_km: float,
    threshold: float = 0.65,
    db_pool: asyncpg.Pool | None = None,
) -> dict:
    import db as _db

    points = _grid_points(lat, lng, radius_km)
    model = load_model()
    sem = asyncio.Semaphore(CONCURRENCY)

    # Normalise grid points to canonical slippy-map tile keys and deduplicate.
    tile_map: dict[tuple[int, int, int], tuple[float, float]] = {}
    for pt in points:
        zxy = _lat_lng_to_tile(pt[0], pt[1])
        if zxy not in tile_map:
            tile_map[zxy] = _tile_center(*zxy)

    unique_tiles = list(tile_map.items())  # [(z,x,y), (tile_lat, tile_lng)]

    # --- Cache lookup ---
    cache_hits: dict[tuple[int, int, int], dict] = {}
    if db_pool is not None:
        cache_hits = await _db.get_cached_tiles(db_pool, [k for k, _ in unique_tiles])

    tiles_to_fetch = [(zxy, coords) for zxy, coords in unique_tiles if zxy not in cache_hits]

    # --- Fetch only uncached tiles ---
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_tile(client, sem, coords[0], coords[1]) for _, coords in tiles_to_fetch]
        fetched_images = await asyncio.gather(*tasks)

    # --- ML inference on fetched tiles ---
    valid_fetched: list[tuple[tuple[int, int, int], tuple[float, float], np.ndarray]] = []
    for (zxy, coords), img in zip(tiles_to_fetch, fetched_images):
        if img is not None:
            valid_fetched.append((zxy, coords, img))

    preds_fetched = np.array([], dtype=np.float32)
    if valid_fetched:
        batch = np.stack([entry[2] for entry in valid_fetched])
        preds_fetched = model.predict(batch, batch_size=16, verbose=0).flatten()

    # --- Persist new tile results to cache ---
    new_cache_entries: list[dict] = []
    for idx, (zxy, coords, _) in enumerate(valid_fetched):
        new_cache_entries.append({
            "z": zxy[0], "x": zxy[1], "y": zxy[2],
            "tile_lat": coords[0], "tile_lng": coords[1],
            "ml_score": float(preds_fetched[idx]),
        })

    if db_pool is not None and new_cache_entries:
        await _db.upsert_tiles(db_pool, new_cache_entries)

    # --- Merge cache hits + live results into unified scored list ---
    fetched_by_zxy: dict[tuple[int, int, int], float] = {
        zxy: float(preds_fetched[idx]) for idx, (zxy, _, _) in enumerate(valid_fetched)
    }

    valid_points: list[tuple[float, float]] = []
    scores_list: list[float] = []

    for zxy, coords in unique_tiles:
        if zxy in cache_hits:
            valid_points.append(coords)
            scores_list.append(cache_hits[zxy]["ml_score"])
        elif zxy in fetched_by_zxy:
            valid_points.append(coords)
            scores_list.append(fetched_by_zxy[zxy])

    if not valid_points:
        return {"detections": [], "tiles_scanned": 0, "tiles_from_cache": 0, "tiles_fetched_live": 0}

    preds = np.array(scores_list, dtype=np.float32)
    neighbor_map = _build_neighbor_map(valid_points, NEIGHBOR_RADIUS_KM)
    prelim = threshold * PRELIM_SCALE
    agg_scores = _aggregate_scores(preds, neighbor_map, prelim)

    candidates = [
        (valid_points[i][0], valid_points[i][1], float(agg_scores[i]))
        for i in range(len(valid_points))
        if preds[i] >= prelim
    ]

    detections = _cluster_detections(candidates, CLUSTER_RADIUS_KM)
    detections = [d for d in detections if d["confidence"] >= threshold]
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # --- Persist clustered detections ---
    if db_pool is not None:
        for d in detections:
            track_id = await _db.upsert_detected_track(db_pool, d["lat"], d["lng"], d["confidence"])
            d["track_id"] = track_id

    return {
        "detections": detections,
        "tiles_scanned": len(valid_points),
        "tiles_from_cache": len(cache_hits),
        "tiles_fetched_live": len(valid_fetched),
    }

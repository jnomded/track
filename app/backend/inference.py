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

# Zoom 16 at 512px gives ~1.2km coverage per tile (radius ~0.61km).
# Grid spacing must be < tile radius so any track always falls well inside a tile.
# At 0.5km, worst-case diagonal offset is 0.35km — track is always near the tile center.
ZOOM = 16
TILE_PX = 512
IMG_SIZE = (224, 224)
GRID_SPACING_KM = 0.5
CONCURRENCY = 16

# Neighbour aggregation: use surrounding tile scores to validate detections
NEIGHBOR_RADIUS_KM = 1.5   # tiles within this distance are considered neighbours
PRELIM_SCALE      = 0.7    # lower gate = threshold * PRELIM_SCALE (e.g. 0.455 at threshold=0.65)
OWN_WEIGHT        = 0.5    # blend weight for tile's own score
NEIGHBOR_WEIGHT   = 0.5    # blend weight for mean neighbour score
NEIGHBOR_FLOOR    = 0.3    # minimum score for a neighbour to contribute to the mean
CLUSTER_RADIUS_KM = 1.2    # detections closer than this are merged into one weighted centroid

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
        # Never penalise: a score can only be boosted by neighbour support, not reduced.
        # This preserves isolated real tracks while still lifting weak edge-of-tile detections.
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


def _validate_track_geometry(img_512: np.ndarray) -> bool:
    """
    Returns True if the 512×512 satellite tile contains an infield shape
    consistent with a 400m running track.

    At zoom 16, 512px tile → ~2.38 m/px (equatorial; ±~5% at lat ≤ 60°).

    Infield size range covers standard (IAAF) through wider non-standard tracks:
        Standard track:    long ≈  66 px (157 m),  short ≈ 31 px ( 73 m), aspect ~2.1
        Wide-curve track:  long ≈  84 px (200 m),  short ≈ 50 px (120 m), aspect ~1.7
        Filter bounds:     long  25–95 px,          short 18–58 px,        aspect 1.2–2.5

    Strategy: detect the green infield blob, validate its size/shape, then
    confirm the surrounding ring is non-green (track surface, not more grass).
    """
    from scipy.ndimage import label, binary_erosion, binary_dilation

    img_f = img_512.astype(np.float32) / 255.0
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    # Green-infield mask: green channel clearly dominates red and blue.
    green_mask = (g > r * 1.08) & (g > b * 1.08) & (g > 0.12)

    # Morphological clean-up: remove isolated noise pixels.
    green_mask = binary_erosion(green_mask, iterations=2)
    green_mask = binary_dilation(green_mask, iterations=3)

    labeled, n_components = label(green_mask)
    if n_components == 0:
        return False

    for comp_id in range(1, n_components + 1):
        comp = labeled == comp_id
        area = int(comp.sum())

        # Area in pixels must be consistent with a 400m track infield.
        if not (250 <= area <= 12_000):
            continue

        rows_idx = np.where(comp.any(axis=1))[0]
        cols_idx = np.where(comp.any(axis=0))[0]
        h = int(rows_idx[-1] - rows_idx[0] + 1)
        w = int(cols_idx[-1] - cols_idx[0] + 1)
        long_px = max(h, w)
        short_px = min(h, w)

        # Bounding-box size check — covers standard through wide-curve tracks.
        if not (25 <= long_px <= 95):
            continue
        if not (18 <= short_px <= 58):
            continue

        # Aspect ratio: standard ~2.1, wide-curve ~1.7; allow generous slack.
        aspect = long_px / short_px
        if not (1.2 <= aspect <= 2.5):
            continue

        # Fill ratio: an oval/stadium blob scores ~0.65–0.90.
        # A plain rectangular grass field scores ~0.95+, rejecting football pitches etc.
        fill_ratio = area / (h * w)
        if not (0.55 <= fill_ratio <= 0.93):
            continue

        # Confirm the ring surrounding the infield is non-green (track surface).
        # Dilate outward by ~6 px (~14 m, about 1.5 lane-widths) to sample the ring.
        ring_outer = binary_dilation(comp, iterations=6)
        ring_mask = ring_outer & ~comp
        ring_pixels = img_f[ring_mask]
        if ring_pixels.shape[0] < 50:
            continue

        ring_r_mean = float(ring_pixels[:, 0].mean())
        ring_g_mean = float(ring_pixels[:, 1].mean())
        # If the ring is also green, this is likely a large grass area, not a track.
        if ring_g_mean > ring_r_mean * 0.98:
            continue

        return True

    return False


async def _fetch_tile(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    lat: float,
    lng: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (img_224, img_512) or None on failure."""
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
            img_orig = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img_512 = np.array(
                img_orig.resize((TILE_PX, TILE_PX), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            img_224 = np.array(
                img_orig.resize(IMG_SIZE, Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            return img_224, img_512
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
    valid_imgs_224: list[np.ndarray] = []
    valid_imgs_512: list[np.ndarray] = []
    for pt, img in zip(points, images):
        if img is not None:
            valid_points.append(pt)
            valid_imgs_224.append(img[0])
            valid_imgs_512.append(img[1])

    if not valid_imgs_224:
        return {"detections": [], "tiles_scanned": 0}

    batch = np.stack(valid_imgs_224)
    preds = model.predict(batch, batch_size=16, verbose=0).flatten()

    neighbor_map = _build_neighbor_map(valid_points, NEIGHBOR_RADIUS_KM)
    prelim = threshold * PRELIM_SCALE
    agg_scores = _aggregate_scores(preds, neighbor_map, prelim)

    candidates = [
        (plat, plng, float(agg_scores[i]))
        for i, (plat, plng) in enumerate(valid_points)
        if preds[i] >= prelim and _validate_track_geometry(valid_imgs_512[i])
    ]

    detections = _cluster_detections(candidates, CLUSTER_RADIUS_KM)
    detections = [d for d in detections if d["confidence"] >= threshold]
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return {"detections": detections, "tiles_scanned": len(valid_imgs_224)}

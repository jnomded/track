import csv
import math
import time
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from pyproj import Transformer
from pystac_client import Client
import planetary_computer as pc
from contextlib import ExitStack


# Configuration


# Approx continental US bounding box 
MIN_LAT, MIN_LON = 24.39, -125.00
MAX_LAT, MAX_LON = 49.38, -66.93

OUTPUT_DIR = Path("dataset_osm")
CHIP_SIZE = 1024  # pixels compatible with MobileNetV2
MAX_POSITIVES = 75 #1200
MAX_NEGATIVES = 75 #1500
NAIP_YEAR = None

# Fraction of positives guaranteed to be stadium-type running tracks (Hayward Field, etc.)
STADIUM_TRACK_RATIO = 0.15
# Fractions of negatives allocated to each hard-negative category
NEG_RATIO_MOTOR_RACING  = 0.05  # NASCAR ovals, F1 circuits, drag strips
NEG_RATIO_SPORTS_STADIUM = 0.15  # football/soccer/baseball stadiums
NEG_RATIO_HORSE_RACING   = 0.05  # horse racing ovals
# Remainder goes to general venues (pitches, golf, sports_centres, etc.)
 
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
#Earth radius in meters for distance calculations
R_EARTH = 6371000.0


@dataclass
class Sample:
    osmid: str
    lat: float
    lon: float
    label: int  # 1=track, 0=not_track
    primary_tag: str
    sport_tag: str
    kind: str


# ----------------------
# Helpers
# ----------------------

def slugify(s: str) -> str:
    """Turn a string into a filename-safe slug."""
    s = s.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def overpass_query(query: str, paused_s: float = 2.5) -> Dict[str, Any]:
    """Execute Overpass API query with a pause to be polite."""
    print("Running Overpass query...")
    resp = requests.post(OVERPASS_URL, data={"data": query})
    resp.raise_for_status()
    time.sleep(paused_s)
    return resp.json()


def parse_overpass_elements(elements: List[Dict], label: int, kind: str) -> List[Sample]:
    """Convert JSON elements to Sample objects."""
    samples = []
    for el in elements:
        # Get coordinates (node vs way/relation)
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        
        if not lat or not lon:
            continue

        tags = el.get("tags", {})
        # Find a primary tag
        primary = next((f"{k}={v}" for k, v in tags.items() if k in {"leisure", "amenity", "landuse"}), "")
        
        samples.append(Sample(
            osmid=f"{el.get('type')}/{el.get('id')}",
            lat=float(lat),
            lon=float(lon),
            label=label,
            primary_tag=primary,
            sport_tag=tags.get("sport", ""),
            kind=kind
        ))
    return samples


def haversine_distance(s1: Sample, s2: Sample) -> float:
    phi1, phi2 = math.radians(s1.lat), math.radians(s2.lat)
    dphi = phi2 - phi1
    dlambda = math.radians(s2.lon - s1.lon)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def filter_by_distance(samples: List[Sample], min_dist_m: float = 100.0) -> List[Sample]:
    """Greedy deduplication to avoid samples being too close."""
    kept = []
    for s in samples:
        if all(haversine_distance(s, k) >= min_dist_m for k in kept):
            kept.append(s)
    return kept


# ----------------------
# Data Fetching
# ----------------------

def get_osm_data() -> Tuple[List[Sample], List[Sample]]:
    box = f"{MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON}"

    # ------------------------------------------------------------------
    # POSITIVES
    # ------------------------------------------------------------------

    # 1a. Priority: stadium-type running tracks (Hayward Field, etc.)
    #     These are leisure=track ways/relations that sit inside an
    #     athletics stadium area.
    print("Fetching stadium running tracks (priority positives)...")
    query_stadium_tracks = f"""
        [out:json][timeout:250][bbox:{box}];
        area["leisure"="stadium"]["sport"~"athletics"]->.stadiums;
        (
          way["leisure"="track"]["sport"~"athletics|running"](area.stadiums);
          relation["leisure"="track"]["sport"~"athletics|running"](area.stadiums);
        );
        out center {int(MAX_POSITIVES * 3)};
    """
    data_stadium = overpass_query(query_stadium_tracks)
    stadium_tracks = parse_overpass_elements(data_stadium.get("elements", []), label=1, kind="stadium_track")

    # 1b. Regular running tracks
    print("Fetching regular running tracks...")
    query_pos = f"""
        [out:json][timeout:250];
        (
          way["leisure"="track"]["sport"~"athletics|running"]({box});
          relation["leisure"="track"]["sport"~"athletics|running"]({box});
        );
        out center {int(MAX_POSITIVES * 5)};
    """
    data_pos = overpass_query(query_pos)
    regular_tracks = parse_overpass_elements(data_pos.get("elements", []), label=1, kind="positive")

    # Deduplicate each group, then deduplicate overlap between them
    stadium_tracks = filter_by_distance(stadium_tracks, 80.0)
    regular_tracks = filter_by_distance(regular_tracks, 80.0)

    stadium_ids = {s.osmid for s in stadium_tracks}
    regular_tracks = [s for s in regular_tracks if s.osmid not in stadium_ids]

    # Guarantee stadium track quota, fill remainder with regular tracks
    max_stadium = int(MAX_POSITIVES * STADIUM_TRACK_RATIO)
    if len(stadium_tracks) > max_stadium:
        stadium_tracks = random.sample(stadium_tracks, max_stadium)
    max_regular = MAX_POSITIVES - len(stadium_tracks)
    if len(regular_tracks) > max_regular:
        regular_tracks = random.sample(regular_tracks, max_regular)

    positives = stadium_tracks + regular_tracks

    # ------------------------------------------------------------------
    # NEGATIVES
    # ------------------------------------------------------------------

    # 2a. General venues (pitches, golf, sports centres, generic stadiums)
    print("Fetching general venue negatives...")
    query_neg = f"""
        [out:json][timeout:1000];
        (
          way["leisure"="stadium"]({box});
          relation["leisure"="stadium"]({box});
          node["leisure"="stadium"]({box});
          way["leisure"="pitch"]({box});
          relation["leisure"="pitch"]({box});
          node["leisure"="pitch"]({box});
          way["leisure"="sports_centre"]({box});
          node["leisure"="sports_centre"]({box});
          way["leisure"="golf_course"]({box});
          node["leisure"="golf_course"]({box});
        );
        out center {int(MAX_NEGATIVES * 5)};
    """
    data_neg = overpass_query(query_neg)
    general_negatives = parse_overpass_elements(data_neg.get("elements", []), label=0, kind="hard_negative")

    # 2b. Motor racing tracks (NASCAR ovals, F1 circuits, drag strips)
    print("Fetching motor racing tracks (negatives)...")
    query_motor = f"""
        [out:json][timeout:250];
        (
          way["leisure"="track"]["sport"~"motor_racing|motorsport"]({box});
          relation["leisure"="track"]["sport"~"motor_racing|motorsport"]({box});
        );
        out center {int(MAX_NEGATIVES * 3)};
    """
    data_motor = overpass_query(query_motor)
    motor_tracks = parse_overpass_elements(data_motor.get("elements", []), label=0, kind="motor_racing")

    # 2c. Sports stadiums (football, soccer, baseball, basketball, hockey)
    print("Fetching sports stadiums (negatives)...")
    query_sports = f"""
        [out:json][timeout:500];
        (
          way["leisure"="stadium"]["sport"~"american_football|soccer|association_football|baseball|basketball|ice_hockey|rugby"]({box});
          relation["leisure"="stadium"]["sport"~"american_football|soccer|association_football|baseball|basketball|ice_hockey|rugby"]({box});
        );
        out center {int(MAX_NEGATIVES * 3)};
    """
    data_sports = overpass_query(query_sports)
    sports_stadiums = parse_overpass_elements(data_sports.get("elements", []), label=0, kind="sports_stadium")

    # 2d. Horse racing tracks
    print("Fetching horse racing tracks (negatives)...")
    query_horse = f"""
        [out:json][timeout:250];
        (
          way["leisure"="track"]["sport"="horse_racing"]({box});
          relation["leisure"="track"]["sport"="horse_racing"]({box});
        );
        out center {int(MAX_NEGATIVES * 3)};
    """
    data_horse = overpass_query(query_horse)
    horse_tracks = parse_overpass_elements(data_horse.get("elements", []), label=0, kind="horse_racing")

    # Deduplicate each negative category independently
    general_negatives = filter_by_distance(general_negatives, 80.0)
    motor_tracks      = filter_by_distance(motor_tracks, 80.0)
    sports_stadiums   = filter_by_distance(sports_stadiums, 80.0)
    horse_tracks      = filter_by_distance(horse_tracks, 80.0)

    # Proportional allocation for negatives
    max_motor  = int(MAX_NEGATIVES * NEG_RATIO_MOTOR_RACING)
    max_sports = int(MAX_NEGATIVES * NEG_RATIO_SPORTS_STADIUM)
    max_horse  = int(MAX_NEGATIVES * NEG_RATIO_HORSE_RACING)

    if len(motor_tracks)    > max_motor:  motor_tracks    = random.sample(motor_tracks, max_motor)
    if len(sports_stadiums) > max_sports: sports_stadiums = random.sample(sports_stadiums, max_sports)
    if len(horse_tracks)    > max_horse:  horse_tracks    = random.sample(horse_tracks, max_horse)

    max_general = MAX_NEGATIVES - len(motor_tracks) - len(sports_stadiums) - len(horse_tracks)
    if len(general_negatives) > max_general:
        general_negatives = random.sample(general_negatives, max_general)

    negatives = general_negatives + motor_tracks + sports_stadiums + horse_tracks

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nPositive breakdown:")
    print(f"  Stadium tracks : {sum(1 for s in positives if s.kind == 'stadium_track')}")
    print(f"  Regular tracks : {sum(1 for s in positives if s.kind == 'positive')}")
    print(f"Negative breakdown:")
    print(f"  General venues : {sum(1 for s in negatives if s.kind == 'hard_negative')}")
    print(f"  Motor racing   : {sum(1 for s in negatives if s.kind == 'motor_racing')}")
    print(f"  Sports stadiums: {sum(1 for s in negatives if s.kind == 'sports_stadium')}")
    print(f"  Horse racing   : {sum(1 for s in negatives if s.kind == 'horse_racing')}")
    print(f"Final counts: {len(positives)} pos, {len(negatives)} neg")

    return positives, negatives


def get_naip_href(item) -> str:
    """Choose the best asset href from a STAC item."""
    # Preferred keys in order
    for k in ["image", "visual", "analytic", "data", "cog"]:
        if k in item.assets:
            return item.assets[k].href
    
    # Fallback to anything that looks like a geotiff
    for _, a in item.assets.items():
        if "geotiff" in (a.media_type or "").lower():
            return a.href
            
    # Fallback to first available
    return next(iter(item.assets.values())).href


def save_naip_chip(sample: Sample, output_path: Path):
    """Fetch NAIP imagery from Planetary Computer and save as PNG. Handles tile boundaries."""
    client = Client.open(PC_STAC_URL)
    
    # 1. Search with a small buffer (approx 1km) to find all relevant tiles
    buf = 0.01  # ~1km buffer in degrees
    bbox = [sample.lon - buf, sample.lat - buf, sample.lon + buf, sample.lat + buf]
    
    search = client.search(
        collections=["naip"],
        bbox=bbox,
        max_items=20, 
    )
    items = list(search.get_items())
    if not items:
        raise ValueError("No NAIP data found")

    # Filter year if specified
    if NAIP_YEAR:
        items = [i for i in items if str(NAIP_YEAR) in (str(i.datetime.year) if i.datetime else i.properties.get("naip:year", ""))]
        if not items:
            raise ValueError(f"No NAIP data for year {NAIP_YEAR}")
    
    # 2. Group by Date to mosaic only consistent imagery
    # Sort by date (newest first)
    items.sort(key=lambda x: x.datetime or "", reverse=True)
    best_date = items[0].datetime.date() if items[0].datetime else items[0].properties.get("naip:year")
    
    # Keep only items from the Best Date (or very close)
    # NAIP acquisition dates for adjacent tiles are usually same day
    relevant_items = []
    for i in items:
        idate = i.datetime.date() if i.datetime else i.properties.get("naip:year")
        if idate == best_date:
            relevant_items.append(i)
            
    if not relevant_items:
        relevant_items = [items[0]]

    # 3. Calculate Target Bounds in the CRS of the first item
    ref_item = pc.sign(relevant_items[0])
    href_ref = get_naip_href(ref_item)
    
    with rasterio.open(href_ref) as ref_ds:
        # Reproject lat/lon center to CRS
        transformer = Transformer.from_crs("EPSG:4326", ref_ds.crs, always_xy=True)
        cx, cy = transformer.transform(sample.lon, sample.lat)
        res_x, res_y = ref_ds.res
        
        # Calculate target bounds for CHIP_SIZE
        # Note: res_y is typically positive in .res, but transform usually has negative dy
        half_w = (CHIP_SIZE * res_x) / 2
        half_h = (CHIP_SIZE * res_y) / 2
        
        # (minx, miny, maxx, maxy)
        target_bounds = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    # 4. Open all relevant files and Merge
    # Use ExitStack to manage multiple open files
    with ExitStack() as stack:
        srcs = []
        for item in relevant_items:
            # We don't need to re-sign ref_item ideally, but pc.sign is cheap
            signed = pc.sign(item)
            href = get_naip_href(signed)
            src = stack.enter_context(rasterio.open(href))
            srcs.append(src)
            
        # Merge with bounds limit
        # This stitches parts of adjacent tiles if needed
        # We assume same CRS/Res for tiles in same area/date
        mosaic, out_trans = merge(srcs, bounds=target_bounds)

    # mosaic is (Bands, Height, Width)
    # Check shape - merge might return slightly different size due to pixel alignment
    _, h, w = mosaic.shape
    if h != CHIP_SIZE or w != CHIP_SIZE:
        # Center crop or pad?
        # Usually checking center is enough if we passed bounds
        # Simple resize (crop to center)
        start_y = (h - CHIP_SIZE) // 2
        start_x = (w - CHIP_SIZE) // 2
        if start_y >= 0 and start_x >= 0:
             mosaic = mosaic[:, start_y:start_y+CHIP_SIZE, start_x:start_x+CHIP_SIZE]
        else:
            # If smaller (rare), pad
             # Not implementing pad for now vs raise
             pass

    # Normalize
    # Handle NAIP which is 4 bands (RGB + NIR)
    # Take first 3 bands
    rgb_bands = mosaic[:3, :, :]
    
    if rgb_bands.dtype == np.uint16:
        rgb_bands = (rgb_bands / 256.0).clip(0, 255).astype(np.uint8)
    elif rgb_bands.dtype != np.uint8:
        m = rgb_bands.max()
        if m > 0:
            rgb_bands = (rgb_bands / m * 255.0).clip(0, 255).astype(np.uint8)

    # Convert to image (H, W, C)
    rgb = np.moveaxis(rgb_bands, 0, -1)
    
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
        
    Image.fromarray(rgb).save(output_path)



def main():

    start_time = time.time()


    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    positives, negatives = get_osm_data()
    all_samples = positives + negatives
    
    # Shuffle to mix processing of positives and negatives
    random.shuffle(all_samples)
    
    if not all_samples:
        print("No samples found. Check your bounding box or connection.")
        return

    csv_rows = []
    print(f"Downloading images for {len(all_samples)} samples ({len(positives)} pos, {len(negatives)} neg)...")
    
    for i, s in enumerate(all_samples):
        cls_name = "track" if s.label == 1 else "not_track"
        folder = OUTPUT_DIR / cls_name
        folder.mkdir(exist_ok=True)
        
        filename = f"{slugify(s.primary_tag or 'unknown')}_{slugify(s.osmid)}.png"
        filepath = folder / filename
        
        # Simple progress
        if i % 10 == 0:
            print(f"[{i+1}/{len(all_samples)}] Processing {s.kind} {s.osmid}...")
        
        try:
            if not filepath.exists():
                save_naip_chip(s, filepath)
            
            csv_rows.append([
                str(filepath.relative_to(OUTPUT_DIR)),
                s.label, s.lat, s.lon, s.osmid, s.primary_tag, s.sport_tag, s.kind
            ])
        except Exception as e:
            if "No NAIP data found" in str(e):
                print(f"  Skipping {s.osmid} (likely outside US NAIP coverage)")
            else:
                print(f"  Error processing {s.osmid}: {e}")

    # Save Metadata
    csv_path = OUTPUT_DIR / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "lat", "lon", "osmid", "primary_tag", "sport", "kind"])
        writer.writerows(csv_rows)
        
    print(f"\nDone! Database created at {OUTPUT_DIR.absolute()}")
    print(f"Total samples: {len(csv_rows)}")
    print(f"Total time elapsed: {time.strftime('%H:%M:%S', time.localtime(time.time() - start_time))}")


if __name__ == "__main__":
    main()

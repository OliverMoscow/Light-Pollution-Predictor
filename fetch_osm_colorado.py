"""
fetch_osm_colorado.py

Fetches commercial businesses from OpenStreetMap via the Overpass API
for a given bounding box. Results are cached to CSV; the cache is reused
as long as the bounding box hasn't changed.

Encoding: the top 100 category/subcategory tags by frequency are assigned
integer indices 0-99 (saved to osm_vocab.json). Everything else gets -1
and should be ignored during training.
"""

import csv
import json
import os
import time
from collections import Counter
import requests
import numpy as np

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS      = {"User-Agent": "dark-sky-research/1.0"}

CACHE_CSV    = "osm_cache.csv"
CACHE_BOUNDS = "osm_cache_bounds.txt"
CACHE_VOCAB  = "osm_vocab.json"

# ---------------------------------------------------------------------------
# Noise exclusion sets
# ---------------------------------------------------------------------------

AMENITY_EXCLUDE = {
    "bench", "waste_basket", "waste_disposal", "recycling",
    "bicycle_parking", "bicycle_repair_station",
    "toilets", "shower", "drinking_water", "watering_place", "fountain",
    "letter_box", "post_box", "vending_machine",
    "shelter", "loading_dock", "parking_entrance",
    "parking", "parking_space", "motorcycle_parking",
    "bbq", "picnic_table", "feeding_place",
    "dog_toilet", "clock", "photo_booth",
    "charging_station", "compressed_air",
    "polling_station", "grave_yard", "crematorium",
    "hunting_stand", "place_of_worship", "atm",
}

LEISURE_EXCLUDE = {
    "pitch", "playground", "park", "garden", "picnic_table",
    "common", "track", "bleachers", "outdoor_seating",
    "dog_park", "slipway", "fishing", "nature_reserve",
    "fitness_station", "bird_hide", "bandstand",
    "swimming_area", "beach_resort", "swimming_pool", "firepit",
}

SHOP_EXCLUDE = {
    "vacant",
}

TOURISM_EXCLUDE = {
    "information", "camp_site", "camp_pitch", "artwork",
    "picnic_site", "viewpoint", "attraction", "caravan_site",
    "chalet", "wilderness_hut", "alpine_hut",
}

BUSINESS_KEYS = ["amenity", "shop", "office", "craft", "leisure", "tourism"]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_query(ql: str, retries: int = 3) -> dict:
    """POST an Overpass QL query and return parsed JSON."""
    for attempt in range(retries):
        print(f"Attempt {attempt + 1} running. This will take a few minutes")
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": ql},
                headers=HEADERS,
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            print(f"  Attempt {attempt + 1} failed ({e}), retrying in 10s...")
            time.sleep(10)


def extract_centroid(element: dict) -> tuple[float, float]:
    """Return (lat, lon) for a node, or the center of a way/relation."""
    if element.get("type") == "node":
        return element.get("lat"), element.get("lon")
    center = element.get("center")
    if center:
        return center.get("lat"), center.get("lon")
    return None, None


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

def load_vocab() -> dict:
    """Return {tag_str: idx} mapping from the saved vocab file."""
    with open(CACHE_VOCAB) as f:
        return json.load(f)


def decode_tag(idx: int, vocab: dict) -> str:
    """Convert a tag index back to its 'category/subcategory' string."""
    idx_to_tag = {v: k for k, v in vocab.items()}
    return idx_to_tag.get(int(idx), "unknown")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cached_bbox() -> str | None:
    """Return the bbox string from the cache, or None if no cache exists."""
    if os.path.exists(CACHE_BOUNDS) and os.path.exists(CACHE_VOCAB):
        with open(CACHE_BOUNDS) as f:
            return f.read().strip()
    return None


def _save_cache(bbox: str, records: list) -> None:
    # Build vocab: top 100 tags by frequency → indices 0-99
    tag_counts = Counter(f"{cat}/{sub}" for _, _, cat, sub, _ in records)
    vocab = {tag: idx for idx, (tag, _) in enumerate(tag_counts.most_common())}

    with open(CACHE_VOCAB, "w") as f:
        json.dump(vocab, f, indent=2)

    with open(CACHE_BOUNDS, "w") as f:
        f.write(bbox)

    with open(CACHE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "tag_idx", "tag_count", "category", "subcategory"])
        for lat, lon, category, subcategory, tag_count in records:
            tag_idx = vocab.get(f"{category}/{subcategory}", -1)
            writer.writerow([lat, lon, tag_idx, tag_count, category, subcategory])

    print(f"Saved {len(records):,} businesses → {CACHE_CSV}  ({len(vocab)} unique tags)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_osm_businesses(bbox: str) -> None:
    """
    Fetch OSM businesses within a bounding box and cache to CSV.

    Skips the Overpass request entirely if the cached bbox matches.
    Load results in main via pandas/numpy using CACHE_CSV and CACHE_VOCAB.

    Parameters
    ----------
    bbox : str
        Overpass-style bounding box: "south,west,north,east"
    """
    if _cached_bbox() == bbox:
        print("Cache hit — bounds unchanged, skipping Overpass fetch.")
        return

    query = f"""
[out:json][timeout:300];
(
  nwr["amenity"]({bbox});
  nwr["shop"]({bbox});
  nwr["office"]({bbox});
  nwr["craft"]({bbox});
  nwr["leisure"]({bbox});
  nwr["tourism"]({bbox});
);
out center tags;
"""
    print(f"Querying Overpass API (bbox={bbox})...", flush=True)
    data = run_query(query)

    records = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue

        t = el.get("tags", {})
        category, subcategory = "", ""
        for k in BUSINESS_KEYS:
            if k in t:
                category, subcategory = k, t[k]
                break

        if not category:
            continue

        if category == "amenity" and subcategory in AMENITY_EXCLUDE:
            continue
        if category == "shop" and subcategory in SHOP_EXCLUDE:
            continue
        if category == "leisure" and subcategory in LEISURE_EXCLUDE:
            continue
        if category == "tourism" and subcategory in TOURISM_EXCLUDE:
            continue

        tag_count = sum(1 for k in BUSINESS_KEYS if k in t)
        records.append([lat, lon, category, subcategory, tag_count])

    _save_cache(bbox, records)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspectOSM(csv_path: str = CACHE_CSV) -> None:
    """Print top 50 OSM tags by count."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df["tag"] = df["category"] + "/" + df["subcategory"]
    counts = df["tag"].value_counts().nlargest(50).sort_values()

    print("=== Top 50 OSM subcategories ===")
    max_count = counts.max()
    for tag, count in counts.items():
        bar = "█" * int(40 * count / max_count)
        print(f"  {tag:<35} {count:>7,}  {bar}")


# ---------------------------------------------------------------------------
# Standalone entry point (Colorado defaults)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    COLORADO_BBOX = "37.0,-109.06,41.0,-102.04"
    t0 = time.perf_counter()
    fetch_osm_businesses(COLORADO_BBOX)
    print(f"Done in {time.perf_counter() - t0:.1f}s")
    inspectOSM()

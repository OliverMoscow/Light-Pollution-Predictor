"""
fetch_osm_colorado.py

Fetches commercial businesses from OpenStreetMap via the Overpass API
for the state of Colorado and saves to osm_businesses_colorado.csv.
"""

import csv
import time
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Colorado bounding box: south, west, north, east
BBOX = "37.0,-109.06,41.0,-102.04"

HEADERS = {"User-Agent": "dark-sky-research/1.0"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_query(ql: str, retries: int = 3) -> dict:
    """POST an Overpass QL query and return parsed JSON."""
    for attempt in range(retries):
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


def extract_centroid(element: dict) -> tuple[float, float] | None:
    """Return (lat, lon) for a node, or the center of a way/relation."""
    etype = element.get("type")
    if etype == "node":
        return element.get("lat"), element.get("lon")
    center = element.get("center")
    if center:
        return center.get("lat"), center.get("lon")
    return None, None


def save_csv(rows: list[dict], filepath: str, fieldnames: list[str]) -> int:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def tags(el: dict) -> dict:
    return el.get("tags", {})


# ---------------------------------------------------------------------------
# Businesses
# ---------------------------------------------------------------------------

# Noise to exclude from amenity — street furniture, infrastructure, natural features
AMENITY_EXCLUDE = {
    "bench", "waste_basket", "waste_disposal", "recycling",
    "bicycle_parking", "bicycle_repair_station",
    "toilets", "shower", "drinking_water", "watering_place", "fountain",
    "letter_box", "post_box", "vending_machine",
    "shelter", "loading_dock", "parking_entrance",
    "bbq", "picnic_table", "feeding_place",
    "dog_toilet", "clock", "photo_booth",
    "charging_station", "compressed_air",
    "polling_station", "grave_yard", "crematorium",
    "hunting_stand",
}

# Noise to exclude from leisure — outdoor recreation with no built lighting
LEISURE_EXCLUDE = {
    "pitch", "playground", "park", "garden", "picnic_table",
    "common", "track", "bleachers", "outdoor_seating",
    "dog_park", "slipway", "fishing", "nature_reserve",
    "fitness_station", "bird_hide", "bandstand",
    "swimming_area", "beach_resort",
}

# Noise to exclude from tourism — non-commercial/outdoor
TOURISM_EXCLUDE = {
    "information", "camp_site", "camp_pitch", "artwork",
    "picnic_site", "viewpoint", "attraction", "caravan_site",
    "chalet", "wilderness_hut", "alpine_hut",
}

BUSINESS_QUERY = f"""
[out:json][timeout:300];
(
  nwr["amenity"]({BBOX});
  nwr["shop"]({BBOX});
  nwr["office"]({BBOX});
  nwr["craft"]({BBOX});
  nwr["leisure"]({BBOX});
  nwr["tourism"]({BBOX});
);
out center tags;
"""

BUSINESS_KEYS = ["amenity", "shop", "office", "craft", "leisure", "tourism"]

BUSINESS_FIELDS = [
    "latitude", "longitude", "name",
    "category", "subcategory",
    "brand", "operator", "opening_hours",
    "addr_street", "addr_housenumber", "addr_city", "addr_postcode",
]


def parse_businesses(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue
        t = tags(el)

        category, subcategory = "", ""
        for k in BUSINESS_KEYS:
            if k in t:
                category = k
                subcategory = t[k]
                break

        if not category:
            continue

        # Skip noise
        if category == "amenity" and subcategory in AMENITY_EXCLUDE:
            continue
        if category == "leisure" and subcategory in LEISURE_EXCLUDE:
            continue
        if category == "tourism" and subcategory in TOURISM_EXCLUDE:
            continue

        rows.append({
            "latitude": lat,
            "longitude": lon,
            "name": t.get("name", t.get("brand", t.get("operator", ""))),
            "category": category,
            "subcategory": subcategory,
            "brand": t.get("brand", ""),
            "operator": t.get("operator", ""),
            "opening_hours": t.get("opening_hours", ""),
            "addr_street": t.get("addr:street", ""),
            "addr_housenumber": t.get("addr:housenumber", ""),
            "addr_city": t.get("addr:city", ""),
            "addr_postcode": t.get("addr:postcode", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching OSM business data for Colorado...\n")
    t0 = time.perf_counter()

    print("Querying Overpass API...", flush=True)
    data = run_query(BUSINESS_QUERY)
    elapsed = time.perf_counter() - t0

    rows = parse_businesses(data)
    count = save_csv(rows, "osm_businesses_colorado.csv", BUSINESS_FIELDS)
    print(f"{count:,} businesses → osm_businesses_colorado.csv  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

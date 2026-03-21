"""
fetch_osm_colorado.py

Fetches light-pollution-relevant data from OpenStreetMap via the Overpass API
for the state of Colorado and saves each category to its own CSV file.

Output files:
  osm_businesses_colorado.csv    - amenity/shop/office/tourism/leisure/craft/healthcare/club
  osm_streetlamps_colorado.csv   - highway=street_lamp nodes
  osm_lit_features_colorado.csv  - any feature tagged lit=*
  osm_landuse_colorado.csv       - land use polygons (commercial/industrial/retail/etc.)
  osm_sports_colorado.csv        - stadiums, pitches, sports centres
  osm_transport_colorado.csv     - airports, rail yards, parking lots
  osm_buildings_colorado.csv     - buildings with building:levels tag
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
                timeout=240,
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
    # ways and relations: Overpass returns a 'center' key when asked
    center = element.get("center")
    if center:
        return center.get("lat"), center.get("lon")
    return None, None


def save_csv(rows: list[dict], filepath: str, fieldnames: list[str]) -> int:
    """Write rows to a CSV file, return the count written."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def tags(el: dict) -> dict:
    return el.get("tags", {})


# ---------------------------------------------------------------------------
# 1. Businesses
# ---------------------------------------------------------------------------

BUSINESS_KEYS = ["amenity", "shop", "office", "tourism", "leisure", "craft", "healthcare", "club"]

BUSINESS_QUERY = f"""
[out:json][timeout:180];
(
  {"".join(f'node["{k}"]({BBOX});way["{k}"]({BBOX});' for k in BUSINESS_KEYS)}
);
out center tags;
"""

BUSINESS_FIELDS = [
    "latitude", "longitude", "entityname",
    "category", "subcategory",
    "cuisine", "brand", "operator", "opening_hours", "is_24_7",
    "phone", "website", "wheelchair", "outdoor_seating", "drive_through", "takeaway",
    "addr_street", "addr_housenumber", "addr_city", "addr_postcode",
]


def parse_businesses(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue
        t = tags(el)

        # Determine category/subcategory from first matching key
        category, subcategory = "", ""
        for k in BUSINESS_KEYS:
            if k in t:
                category = k
                subcategory = t[k]
                break

        name = t.get("name", t.get("brand", t.get("operator", "")))
        opening = t.get("opening_hours", "")

        rows.append({
            "latitude": lat,
            "longitude": lon,
            "entityname": name,
            "category": category,
            "subcategory": subcategory,
            "cuisine": t.get("cuisine", ""),
            "brand": t.get("brand", ""),
            "operator": t.get("operator", ""),
            "opening_hours": opening,
            "is_24_7": "yes" if "24/7" in opening else "",
            "phone": t.get("phone", t.get("contact:phone", "")),
            "website": t.get("website", t.get("contact:website", "")),
            "wheelchair": t.get("wheelchair", ""),
            "outdoor_seating": t.get("outdoor_seating", ""),
            "drive_through": t.get("drive_through", ""),
            "takeaway": t.get("takeaway", ""),
            "addr_street": t.get("addr:street", ""),
            "addr_housenumber": t.get("addr:housenumber", ""),
            "addr_city": t.get("addr:city", ""),
            "addr_postcode": t.get("addr:postcode", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# 2. Street lamps
# ---------------------------------------------------------------------------

LAMP_QUERY = f"""
[out:json][timeout:120];
node["highway"="street_lamp"]({BBOX});
out tags;
"""

LAMP_FIELDS = [
    "latitude", "longitude",
    "lamp_type", "light_colour", "light_direction", "support",
    "ref", "lit",
]


def parse_lamps(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        t = tags(el)
        rows.append({
            "latitude": el.get("lat"),
            "longitude": el.get("lon"),
            "lamp_type": t.get("lamp_type", ""),
            "light_colour": t.get("light:colour", ""),
            "light_direction": t.get("light:direction", ""),
            "support": t.get("support", ""),
            "ref": t.get("ref", ""),
            "lit": t.get("lit", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# 3. Lit features
# ---------------------------------------------------------------------------

LIT_QUERY = f"""
[out:json][timeout:180];
(
  node["lit"]({BBOX});
  way["lit"]({BBOX});
  relation["lit"]({BBOX});
);
out center tags;
"""

LIT_FIELDS = [
    "latitude", "longitude",
    "osm_type", "lit_value",
    "feature_type", "road_type", "name",
]


def parse_lit(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue
        t = tags(el)
        # Determine the primary feature type
        feature_type = (
            t.get("highway") or t.get("amenity") or t.get("landuse") or
            t.get("leisure") or t.get("building") or el.get("type", "")
        )
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "osm_type": el.get("type", ""),
            "lit_value": t.get("lit", ""),
            "feature_type": feature_type,
            "road_type": t.get("highway", ""),
            "name": t.get("name", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. Land use
# ---------------------------------------------------------------------------

LANDUSE_TYPES = [
    "commercial", "retail", "industrial", "residential",
    "greenhouse_horticulture", "farmland", "construction",
]

LANDUSE_QUERY = f"""
[out:json][timeout:180];
(
  {"".join(f'way["landuse"="{lu}"]({BBOX});relation["landuse"="{lu}"]({BBOX});' for lu in LANDUSE_TYPES)}
);
out center tags;
"""

LANDUSE_FIELDS = ["latitude", "longitude", "landuse", "name", "operator"]


def parse_landuse(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue
        t = tags(el)
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "landuse": t.get("landuse", ""),
            "name": t.get("name", ""),
            "operator": t.get("operator", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# 5. Sports facilities
# ---------------------------------------------------------------------------

SPORTS_QUERY = f"""
[out:json][timeout:120];
(
  node["leisure"~"^(stadium|pitch|sports_centre)$"]({BBOX});
  way["leisure"~"^(stadium|pitch|sports_centre)$"]({BBOX});
  relation["leisure"~"^(stadium|pitch|sports_centre)$"]({BBOX});
);
out center tags;
"""

SPORTS_FIELDS = [
    "latitude", "longitude", "name",
    "leisure_type", "sport", "floodlight", "capacity", "operator",
]


def parse_sports(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue
        t = tags(el)
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "name": t.get("name", ""),
            "leisure_type": t.get("leisure", ""),
            "sport": t.get("sport", ""),
            "floodlight": t.get("floodlight", ""),
            "capacity": t.get("capacity", ""),
            "operator": t.get("operator", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# 6. Transport (airports, rail yards, parking)
# ---------------------------------------------------------------------------

TRANSPORT_QUERY = f"""
[out:json][timeout:180];
(
  node["aeroway"="aerodrome"]({BBOX});
  way["aeroway"="aerodrome"]({BBOX});
  relation["aeroway"="aerodrome"]({BBOX});
  node["railway"="yard"]({BBOX});
  way["railway"="yard"]({BBOX});
  node["amenity"="parking"]({BBOX});
  way["amenity"="parking"]({BBOX});
);
out center tags;
"""

TRANSPORT_FIELDS = [
    "latitude", "longitude", "name",
    "transport_type", "subtype",
    "operator", "access", "parking_type",
    "lit", "fee",
]


def parse_transport(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue
        t = tags(el)
        if "aeroway" in t:
            ttype, subtype = "aeroway", t.get("aeroway", "")
        elif "railway" in t:
            ttype, subtype = "railway", t.get("railway", "")
        else:
            ttype, subtype = "parking", t.get("parking", "surface")

        rows.append({
            "latitude": lat,
            "longitude": lon,
            "name": t.get("name", t.get("ref", "")),
            "transport_type": ttype,
            "subtype": subtype,
            "operator": t.get("operator", ""),
            "access": t.get("access", ""),
            "parking_type": t.get("parking", ""),
            "lit": t.get("lit", ""),
            "fee": t.get("fee", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# 7. Buildings with height info
# ---------------------------------------------------------------------------

BUILDINGS_QUERY = f"""
[out:json][timeout:180];
(
  way["building"]["building:levels"]({BBOX});
  relation["building"]["building:levels"]({BBOX});
);
out center tags;
"""

BUILDINGS_FIELDS = [
    "latitude", "longitude", "name",
    "building_type", "levels", "height",
    "operator", "amenity",
]


def parse_buildings(data: dict) -> list[dict]:
    rows = []
    for el in data.get("elements", []):
        lat, lon = extract_centroid(el)
        if lat is None:
            continue
        t = tags(el)
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "name": t.get("name", ""),
            "building_type": t.get("building", ""),
            "levels": t.get("building:levels", ""),
            "height": t.get("height", ""),
            "operator": t.get("operator", ""),
            "amenity": t.get("amenity", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("Businesses",    BUSINESS_QUERY,  parse_businesses, BUSINESS_FIELDS,  "osm_businesses_colorado.csv"),
    ("Street lamps",  LAMP_QUERY,      parse_lamps,      LAMP_FIELDS,      "osm_streetlamps_colorado.csv"),
    ("Lit features",  LIT_QUERY,       parse_lit,        LIT_FIELDS,       "osm_lit_features_colorado.csv"),
    ("Land use",      LANDUSE_QUERY,   parse_landuse,    LANDUSE_FIELDS,   "osm_landuse_colorado.csv"),
    ("Sports",        SPORTS_QUERY,    parse_sports,     SPORTS_FIELDS,    "osm_sports_colorado.csv"),
    ("Transport",     TRANSPORT_QUERY, parse_transport,  TRANSPORT_FIELDS, "osm_transport_colorado.csv"),
    ("Buildings",     BUILDINGS_QUERY, parse_buildings,  BUILDINGS_FIELDS, "osm_buildings_colorado.csv"),
]


def main():
    print("Fetching OSM data for Colorado...\n")
    total_start = time.perf_counter()

    for name, query, parser, fields, outfile in CATEGORIES:
        print(f"[{name}] Querying Overpass API...", flush=True)
        t0 = time.perf_counter()
        data = run_query(query)
        elapsed = time.perf_counter() - t0

        rows = parser(data)
        count = save_csv(rows, outfile, fields)
        print(f"[{name}] {count:,} rows → {outfile}  ({elapsed:.1f}s)\n", flush=True)

        # Be polite to the public Overpass instance
        time.sleep(2)

    print(f"Done. Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()

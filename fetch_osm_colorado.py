"""
fetch_osm_colorado.py

Fetches commercial businesses from OpenStreetMap via the Overpass API
for a given bounding box. Results are cached to CSV; the cache is reused
as long as the bounding box hasn't changed.
"""

import os
import time
import requests
import numpy as np

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS      = {"User-Agent": "dark-sky-research/1.0"}

CACHE_CSV    = "osm_cache.csv"
CACHE_BOUNDS = "osm_cache_bounds.txt"


# Hand-built lookup: (category, subcategory) → light group name.
# Built from the OSM wiki tag list + domain knowledge about light emission.
# Anything not in this table falls through to "other" — we never drop a node.
OSM_TO_LIGHT_GROUP = {
    # -----------------------------------------------------------------------
    # Fuel & automotive — very bright, often 24h canopy lighting
    # -----------------------------------------------------------------------
    ("amenity", "fuel"):                "fuel_station",
    ("amenity", "car_wash"):            "fuel_station",
    ("amenity", "charging_station"):    "fuel_station",   # EV chargers, often lit canopies
    ("shop",    "convenience"):         "fuel_station",   # almost always attached to a gas station
    ("shop",    "gas"):                 "fuel_station",
    ("shop",    "car"):                 "auto_dealer",
    ("shop",    "car_parts"):           "auto_dealer",
    ("shop",    "car_repair"):          "auto_dealer",
    ("shop",    "motorcycle"):          "auto_dealer",
    ("shop",    "tyres"):               "auto_dealer",
    ("shop",    "truck"):               "auto_dealer",
    ("amenity", "car_rental"):          "auto_dealer",
    ("amenity", "car_sharing"):         "auto_dealer",

    # -----------------------------------------------------------------------
    # Food service — lit signs, interior spill
    # -----------------------------------------------------------------------
    ("amenity", "restaurant"):          "food_service",
    ("amenity", "fast_food"):           "food_service",
    ("amenity", "food_court"):          "food_service",
    ("amenity", "cafe"):                "food_service",
    ("amenity", "ice_cream"):           "food_service",
    ("amenity", "bbq"):                 "food_service",
    ("amenity", "snack_bar"):           "food_service",
    ("amenity", "juice_bar"):           "food_service",
    ("amenity", "diner"):               "food_service",
    ("shop",    "bakery"):              "food_service",
    ("shop",    "butcher"):             "food_service",
    ("shop",    "deli"):                "food_service",
    ("shop",    "seafood"):             "food_service",
    ("shop",    "cheese"):              "food_service",
    ("craft",   "bakery"):              "food_service",
    ("craft",   "confectionery"):       "food_service",
    ("craft",   "coffee_roasting"):     "food_service",

    # -----------------------------------------------------------------------
    # Nightlife — bright signage, late hours
    # -----------------------------------------------------------------------
    ("amenity", "bar"):                 "nightlife",
    ("amenity", "pub"):                 "nightlife",
    ("amenity", "nightclub"):           "nightlife",
    ("amenity", "biergarten"):          "nightlife",
    ("amenity", "stripclub"):           "nightlife",
    ("amenity", "adult_gaming_centre"): "nightlife",
    ("shop",    "alcohol"):             "nightlife",      # liquor stores, often late hours
    ("craft",   "brewery"):             "nightlife",
    ("craft",   "winery"):              "nightlife",
    ("craft",   "distillery"):          "nightlife",

    # -----------------------------------------------------------------------
    # Entertainment — stadiums, venues, floodlighting
    # -----------------------------------------------------------------------
    ("leisure", "stadium"):             "entertainment",
    ("leisure", "sports_centre"):       "entertainment",
    ("leisure", "fitness_centre"):      "entertainment",
    ("leisure", "ice_rink"):            "entertainment",
    ("leisure", "bowling_alley"):       "entertainment",
    ("leisure", "amusement_arcade"):    "entertainment",
    ("leisure", "dance"):               "entertainment",
    ("leisure", "water_park"):          "entertainment",
    ("leisure", "escape_game"):         "entertainment",
    ("amenity", "cinema"):              "entertainment",
    ("amenity", "theatre"):             "entertainment",
    ("amenity", "casino"):              "entertainment",
    ("amenity", "arts_centre"):         "entertainment",
    ("amenity", "events_venue"):        "entertainment",
    ("amenity", "music_venue"):         "entertainment",
    ("amenity", "conference_centre"):   "entertainment",
    ("amenity", "exhibition_centre"):   "entertainment",
    ("amenity", "convention_centre"):   "entertainment",
    ("tourism", "museum"):              "entertainment",
    ("tourism", "gallery"):             "entertainment",
    ("tourism", "theme_park"):          "entertainment",
    ("tourism", "zoo"):                 "entertainment",
    ("tourism", "aquarium"):            "entertainment",
    ("tourism", "attraction"):          "entertainment",

    # -----------------------------------------------------------------------
    # Sports (outdoor, potential floodlights) — separate from indoor venues
    # -----------------------------------------------------------------------
    ("leisure", "pitch"):               "sports_outdoor",
    ("leisure", "track"):               "sports_outdoor",
    ("leisure", "golf_course"):         "sports_outdoor",
    ("leisure", "swimming_pool"):       "sports_outdoor",
    ("leisure", "miniature_golf"):      "sports_outdoor",
    ("leisure", "disc_golf_course"):    "sports_outdoor",

    # -----------------------------------------------------------------------
    # Large retail — big box stores, lit parking lots
    # -----------------------------------------------------------------------
    ("shop",    "supermarket"):         "retail_large",
    ("shop",    "mall"):                "retail_large",
    ("shop",    "department_store"):    "retail_large",
    ("shop",    "wholesale"):           "retail_large",
    ("shop",    "warehouse"):           "retail_large",
    ("shop",    "doityourself"):        "retail_large",
    ("shop",    "building_materials"):  "retail_large",
    ("shop",    "garden_centre"):       "retail_large",
    ("shop",    "agrarian"):            "retail_large",

    # -----------------------------------------------------------------------
    # Small retail — storefronts, lower intensity
    # -----------------------------------------------------------------------
    ("shop",    "clothes"):             "retail_small",
    ("shop",    "electronics"):         "retail_small",
    ("shop",    "hardware"):            "retail_small",
    ("shop",    "furniture"):           "retail_small",
    ("shop",    "shoes"):               "retail_small",
    ("shop",    "books"):               "retail_small",
    ("shop",    "music"):               "retail_small",
    ("shop",    "toys"):                "retail_small",
    ("shop",    "sports"):              "retail_small",
    ("shop",    "pet"):                 "retail_small",
    ("shop",    "jewelry"):             "retail_small",
    ("shop",    "watches"):             "retail_small",
    ("shop",    "optician"):            "retail_small",
    ("shop",    "mobile_phone"):        "retail_small",
    ("shop",    "computer"):            "retail_small",
    ("shop",    "hifi"):                "retail_small",
    ("shop",    "photo"):               "retail_small",
    ("shop",    "stationery"):          "retail_small",
    ("shop",    "art"):                 "retail_small",
    ("shop",    "craft"):               "retail_small",
    ("shop",    "gift"):                "retail_small",
    ("shop",    "bag"):                 "retail_small",
    ("shop",    "florist"):             "retail_small",
    ("shop",    "beauty"):              "retail_small",
    ("shop",    "hairdresser"):         "retail_small",
    ("shop",    "cosmetics"):           "retail_small",
    ("shop",    "newsagent"):           "retail_small",
    ("shop",    "kiosk"):               "retail_small",
    ("shop",    "variety_store"):       "retail_small",
    ("shop",    "second_hand"):         "retail_small",
    ("shop",    "antiques"):            "retail_small",
    ("shop",    "video"):               "retail_small",
    ("shop",    "outdoor"):             "retail_small",
    ("shop",    "wine"):                "retail_small",
    ("shop",    "tobacco"):             "retail_small",
    ("shop",    "laundry"):             "retail_small",
    ("shop",    "dry_cleaning"):        "retail_small",
    ("shop",    "tattoo"):              "retail_small",
    ("craft",   "tailor"):              "retail_small",
    ("craft",   "shoemaker"):           "retail_small",
    ("craft",   "jeweller"):            "retail_small",
    ("craft",   "watchmaker"):          "retail_small",

    # -----------------------------------------------------------------------
    # Accommodation — lobby + sign lighting
    # -----------------------------------------------------------------------
    ("tourism", "hotel"):               "accommodation",
    ("tourism", "motel"):               "accommodation",
    ("tourism", "hostel"):              "accommodation",
    ("tourism", "guest_house"):         "accommodation",
    ("tourism", "apartment"):           "accommodation",
    ("tourism", "chalet"):              "accommodation",

    # -----------------------------------------------------------------------
    # Healthcare — 24h lighting for hospitals, pharmacies
    # -----------------------------------------------------------------------
    ("amenity", "hospital"):            "healthcare",
    ("amenity", "pharmacy"):            "healthcare",
    ("amenity", "clinic"):              "healthcare",
    ("amenity", "dentist"):             "healthcare",
    ("amenity", "doctors"):             "healthcare",
    ("amenity", "veterinary"):          "healthcare",
    ("amenity", "social_facility"):     "healthcare",
    ("healthcare", "doctor"):           "healthcare",
    ("healthcare", "dentist"):          "healthcare",
    ("healthcare", "pharmacy"):         "healthcare",
    ("healthcare", "hospital"):         "healthcare",
    ("healthcare", "physiotherapist"):  "healthcare",
    ("healthcare", "optometrist"):      "healthcare",
    ("healthcare", "nurse"):            "healthcare",
    ("healthcare", "alternative"):      "healthcare",

    # -----------------------------------------------------------------------
    # Finance — ATMs, branch windows
    # -----------------------------------------------------------------------
    ("amenity", "bank"):                "finance",
    ("amenity", "atm"):                 "finance",
    ("amenity", "bureau_de_change"):    "finance",
    ("amenity", "money_transfer"):      "finance",
    ("amenity", "payment_centre"):      "finance",
    ("office",  "financial"):           "finance",
    ("office",  "insurance"):           "finance",

    # -----------------------------------------------------------------------
    # Parking — flood-lit surface lots
    # -----------------------------------------------------------------------
    ("amenity", "parking"):             "parking",
    ("amenity", "parking_space"):       "parking",
    ("amenity", "parking_entrance"):    "parking",
    ("amenity", "motorcycle_parking"):  "parking",

    # -----------------------------------------------------------------------
    # Office / commercial buildings
    # -----------------------------------------------------------------------
    ("office",  "company"):             "office",
    ("office",  "government"):          "office",
    ("office",  "lawyer"):              "office",
    ("office",  "accountant"):          "office",
    ("office",  "architect"):           "office",
    ("office",  "engineer"):            "office",
    ("office",  "real_estate"):         "office",
    ("office",  "it"):                  "office",
    ("office",  "research"):            "office",
    ("office",  "telecommunication"):   "office",
    ("office",  "ngo"):                 "office",
    ("office",  "association"):         "office",
    ("office",  "educational_institution"): "office",
    ("office",  "physician"):           "office",
    ("amenity", "post_office"):         "office",
    ("amenity", "courthouse"):          "office",
    ("amenity", "townhall"):            "office",

    # -----------------------------------------------------------------------
    # Industrial — factory/warehouse/utility lighting
    # -----------------------------------------------------------------------
    ("industrial", "factory"):          "industrial",
    ("industrial", "warehouse"):        "industrial",
    ("industrial", "oil_mill"):         "industrial",
    ("industrial", "sawmill"):          "industrial",
    ("industrial", "mine"):             "industrial",
    ("amenity",  "waste_transfer_station"): "industrial",
    ("amenity",  "recycling"):          "industrial",
    ("craft",    "carpenter"):          "industrial",
    ("craft",    "electrician"):        "industrial",
    ("craft",    "plumber"):            "industrial",
    ("craft",    "metal_construction"): "industrial",
    ("craft",    "painter"):            "industrial",

    # -----------------------------------------------------------------------
    # Transport hubs — airports, stations, bus terminals (bright 24h ops)
    # -----------------------------------------------------------------------
    ("aeroway",  "aerodrome"):          "transport_hub",
    ("aeroway",  "terminal"):           "transport_hub",
    ("aeroway",  "hangar"):             "transport_hub",
    ("railway",  "station"):            "transport_hub",
    ("railway",  "yard"):               "transport_hub",
    ("railway",  "halt"):               "transport_hub",
    ("amenity",  "bus_station"):        "transport_hub",
    ("amenity",  "ferry_terminal"):     "transport_hub",
    ("amenity",  "taxi"):               "transport_hub",
    ("amenity",  "bicycle_rental"):     "transport_hub",

    # -----------------------------------------------------------------------
    # Education — campus floodlighting, sports fields, late-night study
    # -----------------------------------------------------------------------
    ("amenity", "university"):          "education",
    ("amenity", "college"):             "education",
    ("amenity", "community_centre"):    "education",
    ("amenity", "library"):             "education",
    ("club",    "sport"):               "education",
    ("club",    "social"):              "education",
    ("club",    "youth"):               "education",

    # -----------------------------------------------------------------------
    # Emergency services — police, fire, EMS (24h exterior lighting)
    # -----------------------------------------------------------------------
    ("amenity", "police"):              "emergency_services",
    ("amenity", "fire_station"):        "emergency_services",
    ("amenity", "ambulance_station"):   "emergency_services",

    # -----------------------------------------------------------------------
    # Low / no emission — explicitly flag these so the model can learn
    # -----------------------------------------------------------------------
    ("amenity", "place_of_worship"):    "low_emission",
    ("amenity", "school"):              "low_emission",
    ("amenity", "kindergarten"):        "low_emission",
    ("amenity", "grave_yard"):          "low_emission",
    ("amenity", "shelter"):             "low_emission",
    ("leisure", "park"):                "low_emission",
    ("leisure", "playground"):          "low_emission",
    ("leisure", "dog_park"):            "low_emission",
    ("leisure", "nature_reserve"):      "low_emission",
    ("leisure", "garden"):              "low_emission",
    ("leisure", "bird_hide"):           "low_emission",
    ("leisure", "fishing"):             "low_emission",
    ("leisure", "horse_riding"):        "low_emission",
    ("tourism", "camp_site"):           "low_emission",
    ("tourism", "caravan_site"):        "low_emission",
    ("tourism", "picnic_site"):         "low_emission",
    ("tourism", "viewpoint"):           "low_emission",
    ("tourism", "wilderness_hut"):      "low_emission",
    ("landuse", "farmland"):            "low_emission",
    ("landuse", "residential"):         "low_emission",
    ("landuse", "grass"):               "low_emission",
    ("landuse", "forest"):              "low_emission",
    ("landuse", "meadow"):              "low_emission",
    ("landuse", "cemetery"):            "low_emission",
    ("landuse", "commercial"):          "retail_large",  # landuse commercial → treat as large retail zone
    ("landuse", "retail"):              "retail_large",
    ("landuse", "industrial"):          "industrial",
    ("landuse", "construction"):        "industrial",
}

# Ordered list of group names. The index becomes the integer token for nn.Embedding.
# "other" is always the last catch-all.
LIGHT_GROUPS = [
    "fuel_station", "auto_dealer", "food_service", "nightlife",
    "entertainment", "sports_outdoor", "retail_large", "retail_small",
    "accommodation", "healthcare", "finance", "parking", "office",
    "industrial", "transport_hub", "education", "emergency_services",
    "low_emission", "other",
]
GROUP_TO_IDX = {g: i for i, g in enumerate(LIGHT_GROUPS)}
NUM_GROUPS   = len(LIGHT_GROUPS)

# ---------------------------------------------------------------------------
# Noise exclusion sets
# ---------------------------------------------------------------------------

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

LEISURE_EXCLUDE = {
    "pitch", "playground", "park", "garden", "picnic_table",
    "common", "track", "bleachers", "outdoor_seating",
    "dog_park", "slipway", "fishing", "nature_reserve",
    "fitness_station", "bird_hide", "bandstand",
    "swimming_area", "beach_resort",
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
        print(f"Attempt {attempt + 1}")
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


def osm_to_lightgroup_idx(category: str, subcategory: str) -> int:
    """Return the integer index for a (category, subcategory) OSM tag pair."""
    name = OSM_TO_LIGHT_GROUP.get((category, subcategory), "other")
    return GROUP_TO_IDX[name]


def decode_lightgroup(idx: int) -> str:
    """Convert an encoded lightgroup integer back to its name."""
    return LIGHT_GROUPS[int(idx)]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cached_bbox() -> str | None:
    """Return the bbox string from the cache, or None if no cache exists."""
    if os.path.exists(CACHE_BOUNDS):
        with open(CACHE_BOUNDS) as f:
            return f.read().strip()
    return None


def _save_cache(bbox: str, records: list) -> None:
    with open(CACHE_BOUNDS, "w") as f:
        f.write(bbox)
    arr = np.array(records, dtype=np.float32)
    np.savetxt(
        CACHE_CSV, arr, delimiter=",",
        header="lat,lon,lightgroup_idx", comments="",
        fmt=["%.6f", "%.6f", "%d"],
    )
    print(f"Saved {len(records):,} businesses → {CACHE_CSV}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_osm_businesses(bbox: str) -> None:
    """
    Fetch OSM businesses within a bounding box and cache to CSV.

    Skips the Overpass request entirely if the cached bbox matches.
    Load the result with load_osm_cache() or decode in main via numpy.

    Parameters
    ----------
    bbox : str
        Overpass-style bounding box: "south,west,north,east"
    """
    if _cached_bbox() == bbox:
        print(f"Cache hit — bounds unchanged, skipping Overpass fetch.")
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
        if category == "leisure" and subcategory in LEISURE_EXCLUDE:
            continue
        if category == "tourism" and subcategory in TOURISM_EXCLUDE:
            continue

        records.append([lat, lon, osm_to_lightgroup_idx(category, subcategory)])

    _save_cache(bbox, records)


# ---------------------------------------------------------------------------
# Standalone entry point (Colorado defaults)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Colorado bounding box: south, west, north, east
    COLORADO_BBOX = "37.0,-109.06,41.0,-102.04"
    t0 = time.perf_counter()
    df = fetch_osm_businesses(COLORADO_BBOX)
    print(f"{len(df):,} businesses fetched in {time.perf_counter() - t0:.1f}s")
    print(df.head())

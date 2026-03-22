"""
Light Pollution Predictor — Colorado
=====================================
Runs two models and compares them:
  1. Linear regression
  2. Graph Attention Network (GNN) with learned embeddings

Both use the same VIIRS labels, OSM businesses, county shapes,
and dark-sky ordinance flags so the R² scores are directly comparable.

Required packages:
    pip install numpy scipy scikit-learn pyshp pyproj matplotlib torch torch-geometric
"""

import csv
import os
import time

import numpy as np
import shapefile
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pyproj import Transformer
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix as _csr_matrix
from sklearn import linear_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
BUSINESSES_FILE = "osm_businesses_colorado.csv"

RADIUS_KM       = 0.5    # linear model: search radius around each pixel
EDGE_RADIUS_M   = 100     # GNN: connect businesses within this many meters
POOL_RADIUS_M   = 400    # GNN: pool businesses within this radius into a pixel
GNN_EPOCHS      = 200     # training epochs for the GNN
GNN_LR          = 5e-5   # learning rate
GNN_PIXELS_PER_EPOCH = 50000  # max training pixels sampled per epoch (prevents OOM)

# ===========================================================================
# MARK: DATA LOADING
# ===========================================================================

def parse_asc(filepath, dtype=np.float32):
    """
    Read an ESRI ASCII raster (.asc).
    Returns the 6-line header dict and the data as a 2D numpy array.
    """
    with open(filepath) as f:
        header = {}
        for _ in range(6):
            parts = f.readline().split()
            header[parts[0].lower()] = float(parts[1])
    data = np.loadtxt(filepath, skiprows=6, dtype=dtype)
    return header, data


def load_businesses(filepath):
    """
    Read the OSM CSV and return parallel arrays of latitudes and longitudes.
    Skips rows with missing or malformed coordinates.
    Also returns the raw rows so the GNN can use extra columns later.
    """
    lats, lons, rows_out = [], [], []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
                rows_out.append(row)
            except (ValueError, KeyError):
                pass
    return np.array(lats), np.array(lons), rows_out


def load_ordinances(filepath):
    """
    Read the county ordinance CSV.
    Returns a dict: county_name -> [fully_shielded, max_limit, floodlights, height_restrictions]
    """
    ordinances = {}
    with open(filepath) as f:
        for row in csv.DictReader(f):
            ordinances[row["Name"]] = [
                int(row["Fully_Shielded"]),
                int(row["Max_Limit"]),
                int(row["Floodlights"]),
                int(row["Height_Restrictions"]),
            ]
    return ordinances


def load_county_data(filepath):
    """
    Read a county shapefile.
    Returns (shapes list, county name list) in matching order.
    """
    sf     = shapefile.Reader(filepath)
    shapes = sf.shapes()
    names  = [rec["NAME"] for rec in sf.iterRecords()]
    return shapes, names


def rasterize_county_grid(shapes, names, header):
    """
    For every VIIRS pixel centre, figure out which county it falls in.
    Returns a 2D array (same shape as VIIRS) where each cell holds the
    county index, or -1 if the pixel is outside all county polygons.
    """
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    xll, yll, cs = header["xllcorner"], header["yllcorner"], header["cellsize"]

    # Build a lat/lon coordinate for every pixel centre
    lats_vec = yll + (nrows - np.arange(nrows) - 0.5) * cs
    lons_vec = xll + (np.arange(ncols) + 0.5) * cs
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)
    pts = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])

    county_grid = np.full(nrows * ncols, -1, dtype=np.int16)

    for i, shape in enumerate(shapes):
        # Fast bounding-box pre-filter before the expensive point-in-polygon test
        bb   = shape.bbox
        mask = ((pts[:, 0] >= bb[0]) & (pts[:, 0] <= bb[2]) &
                (pts[:, 1] >= bb[1]) & (pts[:, 1] <= bb[3]))
        if not mask.any():
            continue
        inside         = np.zeros(len(pts), dtype=bool)
        inside[mask]   = Path(shape.points).contains_points(pts[mask])
        county_grid[inside] = i

    return county_grid.reshape(nrows, ncols)


# ===========================================================================
# MARK: LINEAR REGRESSION
# ===========================================================================

def build_business_score_grid(biz_lats, biz_lons, header):
    """
    For each VIIRS pixel, sum  1/sqrt(dist_km)  for all businesses
    within RADIUS_KM.  Closer businesses contribute more.
    This is a simple scalar "how much commercial activity is nearby?"
    """
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    xll, yll, cs = header["xllcorner"], header["yllcorner"], header["cellsize"]

    lats_vec = yll + (nrows - np.arange(nrows) - 0.5) * cs
    lons_vec = xll + (np.arange(ncols) + 0.5) * cs
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)

    pixel_lats = lat_mesh.ravel()
    pixel_lons = lon_mesh.ravel()

    # Build a KD-tree for fast radius queries (in lat/lon degree space)
    tree        = KDTree(np.column_stack([biz_lats, biz_lons]))
    pixel_pts   = np.column_stack([pixel_lats, pixel_lons])
    radius_deg  = RADIUS_KM / 111.0          # 1 degree ≈ 111 km
    neighbors_list = tree.query_ball_point(pixel_pts, r=radius_deg)

    scores = np.zeros(len(pixel_pts), dtype=np.float32)
    for i, neighbors in enumerate(neighbors_list):
        if not neighbors:
            continue
        dlat       = (biz_lats[neighbors] - pixel_lats[i]) * 111.0
        dlon       = (biz_lons[neighbors] - pixel_lons[i]) * 111.0 * np.cos(np.radians(pixel_lats[i]))
        dist_km    = np.maximum(np.sqrt(dlat**2 + dlon**2), 0.01)
        scores[i]  = np.sum(1.0 / np.sqrt(dist_km))

    return scores.reshape(nrows, ncols)


def build_regression_matrix(viirs_header, viirs_data, biz_scores,
                             county_grid, county_names, ordinances):
    """
    Build the feature matrix X and target vector y for linear regression.
    Each row = one valid VIIRS pixel that has a county + ordinance record.
    Features: [business_score, fully_shielded, max_limit, floodlights, height_restrictions]
    """
    nodata     = viirs_header["nodata_value"]
    rows, cols = np.where(viirs_data != nodata)
    y_all      = viirs_data[rows, cols].astype(np.float32)

    X_rows, valid = [], []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        county_idx  = county_grid[r, c]
        if county_idx < 0:
            continue
        county_name = county_names[county_idx]
        if county_name not in ordinances:
            continue
        X_rows.append([biz_scores[r, c]] + ordinances[county_name])
        valid.append(idx)

    return np.array(X_rows, dtype=np.float32), y_all[valid]


def run_linear_regression(biz_lats, biz_lons, viirs_header, viirs_data,
                          county_grid, county_names, ordinances):
    """
    Fit a plain linear regression and print coefficients + R².
    Returns (model, X, y, y_pred) for the comparison plot.
    """
    print("\n[Baseline] Building business score grid...")
    t0         = time.perf_counter()
    biz_scores = build_business_score_grid(biz_lats, biz_lons, viirs_header)
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    X, y = build_regression_matrix(viirs_header, viirs_data, biz_scores,
                                    county_grid, county_names, ordinances)
    print(f"  X shape: {X.shape},  y shape: {y.shape}")

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    feature_names = ["biz_score", "fully_shielded", "max_limit",
                     "floodlights", "height_restrictions"]
    print("  Coefficients:")
    for name, coef in zip(feature_names, reg.coef_):
        print(f"    {name}: {coef:.6f}")
    print(f"  intercept: {reg.intercept_:.6f}")
    print(f"  R²:        {reg.score(X, y):.4f}")

    y_pred = reg.predict(X)
    return reg, X, y, y_pred


# ===========================================================================
# MARK: OSM TAG ENCODING
# ===========================================================================

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

# Ordered list of group names.  The index in this list becomes the integer
# token fed into nn.Embedding.  "other" is always the last catch-all.
LIGHT_GROUPS   = [
    "fuel_station", "auto_dealer", "food_service", "nightlife",
    "entertainment", "sports_outdoor", "retail_large", "retail_small",
    "accommodation", "healthcare", "finance", "parking", "office",
    "industrial", "transport_hub", "education", "emergency_services",
    "low_emission", "other",
]
GROUP_TO_IDX   = {g: i for i, g in enumerate(LIGHT_GROUPS)}
NUM_GROUPS     = len(LIGHT_GROUPS)


def assign_light_group(row):
    """
    Look up one CSV row's (category, subcategory) in OSM_TO_LIGHT_GROUP.
    Returns the integer index for nn.Embedding.
    Falls back to the "other" index if the tag isn't in the table.
    """
    key = (row.get("category", ""), row.get("subcategory", ""))
    group = OSM_TO_LIGHT_GROUP.get(key, "other")
    return GROUP_TO_IDX[group]


def assign_ordinance_features(row_lat, row_lon, county_grid,
                               county_names, ordinances, header):
    """
    Given a business lat/lon, look up its county and return the 4 ordinance
    binary flags as a list.  Returns [0,0,0,0] if the county has no ordinance.
    """
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    xll, yll, cs = header["xllcorner"], header["yllcorner"], header["cellsize"]

    # Convert lat/lon to raster row/col index
    col = int((row_lon - xll) / cs)
    row = int(nrows - (row_lat - yll) / cs)

    if not (0 <= row < nrows and 0 <= col < ncols):
        return [0, 0, 0, 0]

    county_idx = county_grid[row, col]
    if county_idx < 0:
        return [0, 0, 0, 0]

    county_name = county_names[county_idx]
    return ordinances.get(county_name, [0, 0, 0, 0])


# ===========================================================================
# MARK: GRAPH CONSTRUCTION
# ===========================================================================

_UTM_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)

def assign_ordinance_features_bulk(lats, lons, county_grid,
                                    county_names, ordinances, viirs_header):
    """
    Vectorized version — processes all N businesses at once.
    Returns (N, 4) float32 array.
    """
    # convert lat/lon → grid row/col for all points simultaneously
    origin_x = viirs_header["xllcorner"]
    origin_y = viirs_header["yllcorner"]
    pixel_w = viirs_header["cellsize"]
    pixel_h = viirs_header["cellsize"]


    cols = ((lons - origin_x) / pixel_w).astype(np.int32)
    rows = ((origin_y - lats) / abs(pixel_h)).astype(np.int32)

    # clip to grid bounds — businesses outside the raster get index 0
    rows = np.clip(rows, 0, county_grid.shape[0] - 1)
    cols = np.clip(cols, 0, county_grid.shape[1] - 1)

    # single fancy-index lookup — all N businesses at once
    county_ids = county_grid[rows, cols]   # shape (N,)

    # vectorized ordinance flag lookup using numpy arrays
    # pre-build ordinance arrays indexed by county_id
    ord_array = np.zeros((len(county_names) + 1, 4), dtype=np.float32)
    for idx, name in enumerate(county_names):
        if name in ordinances:
            ord_array[idx] = ordinances[name]   # 4 flags per county

    return ord_array[county_ids]   # shape (N, 4) — one lookup, done


def build_graph(biz_rows, biz_lats, biz_lons,
                county_grid, county_names, ordinances, viirs_header):

    n = len(biz_rows)
    print(f"  Building graph from {n:,} businesses...")

    # 1. light group ids — list comprehension is fine here since
    #    assign_light_group is just a dict lookup, already fast
    group_ids = np.array(
        [assign_light_group(r) for r in biz_rows], dtype=np.int64
    )

    # 2. ordinance features — bulk vectorized raster lookup
    ord_feats = assign_ordinance_features_bulk(
        biz_lats, biz_lons,
        county_grid, county_names, ordinances, viirs_header
    )

    # 3. UTM projection — vectorized, transformer created once at module level
    biz_x, biz_y = _UTM_TRANSFORMER.transform(biz_lons, biz_lats)
    biz_utm = np.column_stack([biz_x, biz_y]).astype(np.float32)

    # 4. KD-tree edges — no Python loop
    tree  = cKDTree(biz_utm)
    pairs = tree.query_pairs(r=EDGE_RADIUS_M, output_type="ndarray")

    if len(pairs) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr  = torch.zeros(0, 2, dtype=torch.float)
    else:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        dists = np.linalg.norm(
            biz_utm[i_idx] - biz_utm[j_idx], axis=1
        ).astype(np.float32)

        src        = np.concatenate([i_idx, j_idx])
        dst        = np.concatenate([j_idx, i_idx])
        edge_dists = np.concatenate([dists, dists])

        sigma        = EDGE_RADIUS_M / 2.0
        norm_dist    = edge_dists / EDGE_RADIUS_M
        gauss_weight = np.exp(-(edge_dists**2) / (2 * sigma**2))

        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr  = torch.tensor(
            np.column_stack([norm_dist, gauss_weight]), dtype=torch.float
        )

    graph = Data(
        group_ids  = torch.from_numpy(group_ids),
        ord_feats  = torch.from_numpy(ord_feats),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        pos        = torch.from_numpy(biz_utm),
    )

    print(f"  Nodes: {graph.num_nodes:,}   Edges: {graph.num_edges:,}")
    return graph


# ===========================================================================
# MARK: GNN MODEL
# ===========================================================================

class LightPollutionGNN(nn.Module):
    """
    Three-stage model:
      1. Embed each business node (NAICS/light-group category → 16d vector,
         concatenated with 4 ordinance flags → 20d input per node)
      2. Three GAT layers propagate information across the spatial graph
         so each node learns about its commercial neighbourhood
      3. For each VIIRS pixel, pool nearby node embeddings with Gaussian
         distance weighting, then push through a small MLP to predict radiance
    """

    def __init__(self, num_groups=NUM_GROUPS, embed_dim=16, hidden_dim=32, dropout=0.2):
        super().__init__()

        # Stage 1 — learned embedding for each light group
        # Similar categories end up close in embedding space automatically
        self.embedding = nn.Embedding(num_groups, embed_dim)

        # The full node input = embedding (16d) + ordinance flags (4d) = 20d
        # Embedding value can be changed to reflect the number of categories. right now we have around 19 categories so 16 works well for that.
        node_in_dim = embed_dim + 4

        # Stage 2 — three GAT layers
        # Each GATConv layer passes values between business in order to get an 
        # encoding of the business which not just stores the category of that 
        # business but also information about what "neighborhood" the business is in.
        # See more https://distill.pub/2021/gnn-intro/

        self.gat1 = GATConv(node_in_dim, hidden_dim // 2,
                            heads=2, edge_dim=2, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim,  hidden_dim // 2,
                            heads=2, edge_dim=2, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_dim,  hidden_dim // 2,
                            heads=2, edge_dim=2, dropout=dropout, concat=True)

        # Stage 3 — prediction head (runs once per pixel, not per node)
        # Input = pooled 32d embedding  →  scalar radiance output
        # Was 62 but changed to conserve ram
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode_nodes(self, graph):
        """
        Run the embedding + GAT layers over all business nodes.
        Returns node_emb: (N, hidden_dim) tensor — one vector per business.
        """
        # Embed the light group index into a 16d vector
        emb = self.embedding(graph.group_ids)           # (N, 16)

        # Concatenate the 4 ordinance flags onto each node's embedding
        node_feats = torch.cat([emb, graph.ord_feats], dim=1)   # (N, 20)

        # Three rounds of neighbourhood message passing
        h = F.relu(self.gat1(node_feats, graph.edge_index, graph.edge_attr))
        h = F.relu(self.gat2(h,          graph.edge_index, graph.edge_attr))
        h = F.relu(self.gat3(h,          graph.edge_index, graph.edge_attr))
        return h    # (N, 64)

    def predict_pixels(self, node_emb, node_pos_np, pixel_coords_np):
        """
        Vectorized pooling via a sparse weight matrix.

        All distances and weights are computed in numpy (outside autograd),
        then assembled into a single sparse (P x N) matrix.  One matmul
        W @ node_emb gives the pooled embeddings with a tiny, clean
        computation graph — avoids the O(P) autograd nodes from the old loop.
        """
        P = len(pixel_coords_np)
        N = len(node_pos_np)
        sigma = POOL_RADIUS_M / 2.0

        tree = cKDTree(node_pos_np)
        nbrs_list = tree.query_ball_point(pixel_coords_np, r=POOL_RADIUS_M)

        row_idx, col_idx, weights = [], [], []
        for i, nbrs in enumerate(nbrs_list):
            if not nbrs:
                continue
            nbrs = np.array(nbrs, dtype=np.int64)
            dists = np.linalg.norm(node_pos_np[nbrs] - pixel_coords_np[i], axis=1)
            w = np.exp(-(dists ** 2) / (2 * sigma ** 2))
            w /= w.sum()
            row_idx.append(np.full(len(nbrs), i, dtype=np.int64))
            col_idx.append(nbrs)
            weights.append(w.astype(np.float32))

        if row_idx:
            rows = np.concatenate(row_idx)
            cols = np.concatenate(col_idx)
            vals = np.concatenate(weights)
            W = torch.sparse_coo_tensor(
                torch.tensor(np.stack([rows, cols]), dtype=torch.long),
                torch.tensor(vals, dtype=torch.float, device=node_emb.device),
                (P, N),
            ).coalesce()
            pooled = torch.sparse.mm(W, node_emb)   # (P, hidden_dim)
        else:
            pooled = torch.zeros(P, node_emb.shape[1], device=node_emb.device)

        raw = self.head(pooled).squeeze(1)
        return raw

    def predict_pixels_W(self, node_emb, W_sparse, P):
        """
        Pooling given a precomputed sparse weight matrix W_sparse (P x N).
        Avoids rebuilding the KDTree and weight matrix every epoch.
        """
        if W_sparse is not None:
            pooled = torch.sparse.mm(W_sparse, node_emb)   # (P, hidden_dim)
        else:
            pooled = torch.zeros(P, node_emb.shape[1], device=node_emb.device)
        raw = self.head(pooled).squeeze(1)
        return raw

    def forward(self, graph, pixel_coords_np):
        node_emb     = self.encode_nodes(graph)
        node_pos_np  = graph.pos.detach().cpu().numpy()
        return self.predict_pixels(node_emb, node_pos_np, pixel_coords_np)


# ===========================================================================
# MARK: GNN TRAINING
# ===========================================================================

def get_pixel_coords_utm(viirs_header, viirs_data):
    nrows  = int(viirs_header["nrows"])
    ncols  = int(viirs_header["ncols"])
    xll    = viirs_header["xllcorner"]
    yll    = viirs_header["yllcorner"]
    cs     = viirs_header["cellsize"]
    nodata = viirs_header["nodata_value"]

    lats_vec = yll + (nrows - np.arange(nrows) - 0.5) * cs
    lons_vec = xll + (np.arange(ncols) + 0.5) * cs
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)

    valid_mask = viirs_data != nodata
    pix_lats   = lat_mesh[valid_mask]
    pix_lons   = lon_mesh[valid_mask]
    radiance   = viirs_data[valid_mask].astype(np.float32)

    #need to mask the coords because raster is bigger than just colorado
    co_mask  = (
        (pix_lons >= -109.06) & (pix_lons <= -102.04) &
        (pix_lats >=   36.99) & (pix_lats <=   41.01)
    )
    pix_lats = pix_lats[co_mask]
    pix_lons = pix_lons[co_mask]
    radiance = radiance[co_mask]
    print(f"  Pixels after Colorado clip: {len(radiance):,}")

    # Convert to UTM metres for consistent distance calculations
    transformer  = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)
    pix_x, pix_y = transformer.transform(pix_lons, pix_lats)
    pixel_utm    = np.column_stack([pix_x, pix_y]).astype(np.float32)

    return pixel_utm, radiance, pix_lons, pix_lats


def spatial_train_val_split(pix_lons, pix_lats, val_fraction=0.2, block_deg=0.5):
    """
    Split pixels into train and validation sets using SPATIAL blocks.

    Why not random split?  Nearby pixels share similar radiance values
    (spatial autocorrelation).  A random split leaks nearby pixels into
    the val set which makes R² look better than it really is.

    Instead we hold out geographically separate 0.5° × 0.5° blocks.
    """
    block_x = (pix_lons // block_deg).astype(int)
    block_y = (pix_lats // block_deg).astype(int)
    block_ids = block_x * 1000 + block_y

    unique_blocks = np.unique(block_ids)
    np.random.shuffle(unique_blocks)
    n_val   = max(1, int(len(unique_blocks) * val_fraction))
    val_blk = set(unique_blocks[:n_val])

    val_mask   = np.array([b in val_blk for b in block_ids])
    train_mask = ~val_mask
    return train_mask, val_mask


def run_gnn(graph, viirs_header, viirs_data):
    """
    Train the GNN and return (model, pixel_utm, radiance, y_pred).
    Prints train/val loss every 10 epochs.
    """
    print("\n[GNN] Preparing pixel coordinates...")
    pixel_utm, radiance, pix_lons, pix_lats = get_pixel_coords_utm(
        viirs_header, viirs_data
    )
    print(f"  {len(radiance):,} valid pixels")
    print(f"Radiance stats:")
    print(f"  min:    {radiance.min():.4f}")
    print(f"  max:    {radiance.max():.4f}")
    print(f"  mean:   {radiance.mean():.4f}")
    print(f"  median: {np.median(radiance):.4f}")
    print(f"  >0:     {(radiance > 0).sum():,} ({100*(radiance>0).mean():.1f}%)")
    print(f"  >1:     {(radiance > 1).sum():,} ({100*(radiance>1).mean():.1f}%)")

    # Fix 1 — log-transform the target so the model learns across the full range
    radiance_raw = radiance.copy()
    radiance     = np.log(radiance + 1e-4)

    # Fix 2 — keep only pixels bright enough to be informative
    bright_enough = radiance > np.log(0.5 + 1e-4)
    pixel_utm = pixel_utm[bright_enough]
    radiance  = radiance[bright_enough]
    pix_lons  = pix_lons[bright_enough]
    pix_lats  = pix_lats[bright_enough]
    radiance_raw = radiance_raw[bright_enough]
    print(f"  Pixels after brightness filter: {len(radiance):,}")

    print(f"  Graph nodes: {graph.num_nodes:,}  edges: {graph.num_edges:,}")
    node_mb  = graph.group_ids.nbytes + graph.ord_feats.nbytes + graph.pos.nbytes
    edge_mb  = graph.edge_index.nbytes + graph.edge_attr.nbytes
    print(f"  Graph tensor mem: nodes={node_mb/1e6:.1f} MB  edges={edge_mb/1e6:.1f} MB")

    # Spatial train / val split
    train_mask, val_mask = spatial_train_val_split(pix_lons, pix_lats)
    print(f"  Train pixels: {train_mask.sum():,}   Val pixels: {val_mask.sum():,}")

    y_tensor = torch.tensor(radiance, dtype=torch.float)

    model     = LightPollutionGNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=GNN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    best_weights  = None

    # Precompute pixel→node weight matrix once so encode_nodes isn't called
    # twice per epoch (was 6 GAT passes/epoch; now 2)
    print("\n[GNN] Precomputing pixel-node weight matrix...")
    t_w          = time.perf_counter()
    _node_pos_np = graph.pos.detach().cpu().numpy()
    _N           = len(_node_pos_np)
    _P           = len(pixel_utm)
    _sigma       = POOL_RADIUS_M / 2.0
    _tree        = cKDTree(_node_pos_np)
    BATCH          = 100_000
    _rr, _cc, _vv  = [], [], []
    _has_biz_count = 0

    for start in range(0, _P, BATCH):
        end        = min(start + BATCH, _P)
        batch_pts  = pixel_utm[start:end]
        nbrs_batch = _tree.query_ball_point(batch_pts, r=POOL_RADIUS_M)

        for local_i, _nbrs in enumerate(nbrs_batch):
            if not _nbrs:
                continue
            _has_biz_count += 1
            i   = start + local_i
            _nb = np.array(_nbrs, dtype=np.int64)
            _d  = np.linalg.norm(_node_pos_np[_nb] - pixel_utm[i], axis=1)
            _w  = np.exp(-(_d ** 2) / (2 * _sigma ** 2))
            _w /= _w.sum()
            _rr.append(np.full(len(_nb), i, dtype=np.int64))
            _cc.append(_nb)
            _vv.append(_w.astype(np.float32))

        if (start // BATCH) % 10 == 0:
            print(f"  {end:,} / {_P:,} pixels processed...", flush=True)

    if _rr:
        _rr = np.concatenate(_rr); _cc = np.concatenate(_cc); _vv = np.concatenate(_vv)
        W_scipy = _csr_matrix((_vv, (_rr, _cc)), shape=(_P, _N))
    else:
        W_scipy = _csr_matrix((_P, _N))
    print(f"  done ({time.perf_counter() - t_w:.1f}s)  nnz={W_scipy.nnz:,}")
    print(f"  Pixels with businesses in pool radius: "
          f"{_has_biz_count:,} / {_P:,} ({100 * _has_biz_count / _P:.1f}%)")
    print("=== Coordinate sanity check ===")
    print(f"Pixel UTM range:")
    print(f"  X: {pixel_utm[:, 0].min():.0f} → {pixel_utm[:, 0].max():.0f}")
    print(f"  Y: {pixel_utm[:, 1].min():.0f} → {pixel_utm[:, 1].max():.0f}")

    node_pos = graph.pos.numpy()
    print(f"Business UTM range:")
    print(f"  X: {node_pos[:, 0].min():.0f} → {node_pos[:, 0].max():.0f}")
    print(f"  Y: {node_pos[:, 1].min():.0f} → {node_pos[:, 1].max():.0f}")

    def make_W_torch(idx):
        """Slice W_scipy rows for idx and return a PyTorch sparse COO tensor."""
        sub = W_scipy[idx].tocoo()
        if sub.nnz == 0:
            return None
        indices = torch.tensor(
            np.stack([sub.row.astype(np.int64), sub.col.astype(np.int64)]),
            dtype=torch.long,
        )
        values = torch.tensor(sub.data.astype(np.float32), dtype=torch.float)
        return torch.sparse_coo_tensor(indices, values, (len(idx), _N)).coalesce()

    # precompute training split. I was running into an issue where it was seeing too many pixels 
    # without any busineses which meant it had no data to go off for those pixels.
    train_all     = np.where(train_mask)[0]
    row_has_nnz   = np.diff(W_scipy.indptr) > 0   # vectorized, no Python loop
    train_has_biz = row_has_nnz[train_all]
    train_with_biz = train_all[train_has_biz]
    train_no_biz   = train_all[~train_has_biz]
    print(f"  Train pixels with businesses: {len(train_with_biz):,}")
    print(f"  Train pixels without:         {len(train_no_biz):,}")

    # precompute val split the same way as train
    val_all       = np.where(val_mask)[0]
    val_has_biz   = row_has_nnz[val_all]
    val_with_biz  = val_all[val_has_biz]
    val_no_biz    = val_all[~val_has_biz]
    print(f"  Val pixels with businesses:   {len(val_with_biz):,}")
    print(f"  Val pixels without:           {len(val_no_biz):,}")

    val_losses = []

    print(f"\n[GNN] Training for {GNN_EPOCHS} epochs...")
    for epoch in range(1, GNN_EPOCHS + 1):

        # use all business pixels every epoch — there are only 19k of them
        # fill remaining slots with dark pixels
        n_biz  = len(train_with_biz)
        n_dark = min(GNN_PIXELS_PER_EPOCH - n_biz, len(train_no_biz))
        train_idx = np.concatenate([
            train_with_biz,                                        
            np.random.choice(train_no_biz, n_dark, replace=False), 
        ])

        # val sampling — same 80/20 split as train
        n_val_biz  = len(val_with_biz)
        n_val_dark = min(GNN_PIXELS_PER_EPOCH - n_val_biz, len(val_no_biz))
        val_idx    = np.concatenate([
            val_with_biz,
            np.random.choice(val_no_biz, n_val_dark, replace=False),
        ])

        # --- encode once per epoch (train mode, dropout active) ---
        model.train()
        optimiser.zero_grad()
        node_emb     = model.encode_nodes(graph)
        W_train      = make_W_torch(train_idx)
        y_pred_train = model.predict_pixels_W(node_emb, W_train, len(train_idx))
        train_loss   = F.mse_loss(y_pred_train, y_tensor[train_idx])
        train_loss.backward()
        optimiser.step()

        # --- validation pass — reuse node_emb, no second encode ---
        model.eval()
        with torch.no_grad():
            W_val      = make_W_torch(val_idx)
            y_pred_val = model.predict_pixels_W(
                node_emb.detach(), W_val, len(val_idx)
            )
            val_loss = F.mse_loss(y_pred_val, y_tensor[val_idx])

            y_true_orig = torch.exp(y_tensor[val_idx]) - 1e-4
            y_pred_orig = torch.exp(y_pred_val)        - 1e-4
            ss_res = ((y_true_orig - y_pred_orig) ** 2).sum()
            ss_tot = ((y_true_orig - y_true_orig.mean()) ** 2).sum()
            r2     = (1 - ss_res / ss_tot).item()

        print(f"  Epoch {epoch:3d} | train MSE {train_loss.item():.4f} "
            f"| val MSE {val_loss.item():.4f} | val R² {r2:.4f}", flush=True)

        val_losses.append(val_loss.item())
        if epoch > 1:
            improvement_rate = (val_losses[0] - val_loss.item()) / epoch
            epochs_to_zero   = val_loss.item() / improvement_rate
            print(f"  → improving {improvement_rate:.5f}/epoch, "
                  f"~{epochs_to_zero:.0f} epochs to cross zero R²", flush=True)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}

        del node_emb, y_pred_train, W_train, y_pred_val, W_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Restore best weights and compute final predictions over ALL pixels
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        node_emb_final = model.encode_nodes(graph)
        W_all = make_W_torch(np.arange(len(pixel_utm)))
        y_pred_all_log = model.predict_pixels_W(
            node_emb_final, W_all, len(pixel_utm)
        ).numpy()

    # Convert log-scale predictions back to original radiance scale
    y_pred_all = np.exp(y_pred_all_log) - 1e-4

    # R² over the validation set on original (linear) scale
    y_val   = radiance_raw[val_mask]
    yp_val  = y_pred_all[val_mask]
    ss_res  = np.sum((y_val - yp_val)**2)
    ss_tot  = np.sum((y_val - y_val.mean())**2)
    r2_val  = 1 - ss_res / ss_tot
    print(f"\n[GNN] Spatial-CV R²: {r2_val:.4f}  (best val MSE: {best_val_loss:.4f})")

    return model, pixel_utm, radiance_raw, y_pred_all, r2_val


# ===========================================================================
# MARK: COMPARISON PLOT
# ===========================================================================

def plot_comparison(y_lin, yp_lin, r2_lin,
                    y_gnn, yp_gnn, r2_gnn):
    """
    Side-by-side scatter plots: actual vs predicted radiance.
    Left = linear regression baseline, right = GNN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, y, yp, title, r2 in [
        (axes[0], y_lin, yp_lin, "Linear regression (baseline)", r2_lin),
        (axes[1], y_gnn, yp_gnn, "GNN (GAT + embeddings)",       r2_gnn),
    ]:
        ax.scatter(y, yp, alpha=0.2, s=4)
        lims = [min(y.min(), yp.min()), max(y.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual VIIRS radiance")
        ax.set_ylabel("Predicted VIIRS radiance")
        ax.set_title(f"{title}\nR² = {r2:.4f}")

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "comparison_plot.png"), dpi=150)
    print("\nPlot saved to comparison_plot.png")
    plt.show()


# ===========================================================================
# MAIN — runs both models end-to-end and compares them
# ===========================================================================

def main():
    # ---- Load shared data ------------------------------------------------
    print("Loading VIIRS raster...")
    viirs_header, viirs_data = parse_asc(
        os.path.join(BASE_DIR, "colorado_2023_viirs.asc")
    )

    print("Loading county shapefile...")
    shapes, county_names = load_county_data(
        os.path.join(BASE_DIR, "Counties_colorado_2005")
    )

    print("Loading dark-sky ordinances...")
    ordinances = load_ordinances(
        os.path.join(BASE_DIR, "colorado_dark_sky_ordinances.csv")
    )

    print("Rasterizing county grid (slow — runs once)...")
    t0          = time.perf_counter()
    county_grid = rasterize_county_grid(shapes, county_names, viirs_header)
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    businesses_path = os.path.join(BASE_DIR, BUSINESSES_FILE)
    print("Loading OSM businesses...")
    biz_lats, biz_lons, biz_rows = load_businesses(businesses_path)
    print(f"  {len(biz_lats):,} businesses loaded")

    # ---- Model 1: Linear regression baseline -----------------------------
    reg, X_lin, y_lin, yp_lin = run_linear_regression(
        biz_lats, biz_lons, viirs_header, viirs_data,
        county_grid, county_names, ordinances
    )
    r2_lin = reg.score(X_lin, y_lin)

    # ---- Model 2: GNN ----------------------------------------------------
    print("\n[GNN] Building spatial business graph...")
    t0    = time.perf_counter()
    graph = build_graph(biz_rows, biz_lats, biz_lons,
                        county_grid, county_names, ordinances, viirs_header)
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    degree = torch.zeros(graph.num_nodes, dtype=torch.long)
    degree.scatter_add_(0, graph.edge_index[0], 
                        torch.ones(graph.edge_index.shape[1], dtype=torch.long))

    print(f"Total edges: {graph.num_edges:,}")
    print(f"Isolated nodes (no edges): {(degree == 0).sum():,} "
        f"/ {graph.num_nodes:,} "
        f"({100*(degree==0).float().mean():.1f}%)")
    print(f"Mean degree: {degree.float().mean():.2f}")
    print(f"Max degree:  {degree.max().item()}")

    model, pixel_utm, y_gnn, yp_gnn, r2_gnn = run_gnn(
        graph, viirs_header, viirs_data
    )

    # ---- Compare ---------------------------------------------------------
    print("\n========== RESULTS ==========")
    print(f"  Linear regression R²: {r2_lin:.4f}")
    print(f"  GNN spatial-CV R²:    {r2_gnn:.4f}")
    print(f"  Improvement:          {r2_gnn - r2_lin:+.4f}")
    print("=" * 30)

    plot_comparison(y_lin, yp_lin, r2_lin,
                    y_gnn, yp_gnn, r2_gnn)


if __name__ == "__main__":
    main()
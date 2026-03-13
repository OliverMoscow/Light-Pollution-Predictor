import csv
import os
import time
import numpy as np
import shapefile
from pyproj import Transformer
from sklearn import linear_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_viirs_data = None
_nlcd_data = None
_county_data = None
_ordinances = None
_businesses = None  # dict: "lats", "lons" (np arrays), "names" (list)
# Converts lat/lon (EPSG:4326) to Albers Equal Area meters (EPSG:5070),
# which is the projected coordinate system the NLCD raster uses.
_to_albers = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

RADIUS_FT = 250
RADIUS_KM = RADIUS_FT * 0.0003048  # ~0.0762 km

# All NLCD 2016 land cover class codes
NLCD_CLASSES = [11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]

FEATURE_NAMES = (
    [f"nlcd_{c}" for c in NLCD_CLASSES]
    + ["business_count"]
    + ["fully_shielded", "max_limit", "floodlights", "height_restrictions"]
)


def _parse_asc(filepath, dtype=np.float32):
    with open(filepath) as f:
        header = {}
        for _ in range(6):
            parts = f.readline().split()
            header[parts[0].lower()] = float(parts[1])
    data = np.loadtxt(filepath, skiprows=6, dtype=dtype)
    return header, data


def _grid_lookup(header, data, lat, lon):
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    cellsize = header["cellsize"]
    nodata = header["nodata_value"]

    col = int((lon - xll) / cellsize)
    row = int((yll + nrows * cellsize - lat) / cellsize)

    if not (0 <= row < nrows and 0 <= col < ncols):
        return None

    val = data[row][col]
    if val == nodata:
        return None
    return val

def _point_in_polygon(x, y, points, parts):
    inside = False
    for i, start in enumerate(parts):
        end = parts[i + 1] if i + 1 < len(parts) else len(points)
        ring = points[start:end]
        n = len(ring)
        j = n - 1
        for k in range(n):
            xi, yi = ring[k]
            xj, yj = ring[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = k
    return inside


def _load_county_data(filepath):
    sf = shapefile.Reader(filepath)
    shapes = sf.shapes()
    names = [rec["NAME"] for rec in sf.iterRecords()]
    return shapes, names


def _find_county(shapes, names, lat, lon):
    for i, s in enumerate(shapes):
        bbox = s.bbox
        if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
            continue
        if _point_in_polygon(lon, lat, s.points, s.parts):
            return names[i]
    return None


def _load_businesses(filepath):
    names, lats, lons = [], [], []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
                names.append(row["entityname"])
            except (ValueError, KeyError):
                pass  # skip rows with missing/invalid coords
    return {"lats": np.array(lats), "lons": np.array(lons), "names": names}


def _haversine_km(lat, lon, lats_arr, lons_arr):
    """Vectorized haversine distance from (lat, lon) to arrays of coords."""
    dlat = np.radians(lats_arr - lat)
    dlon = np.radians(lons_arr - lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat)) * np.cos(np.radians(lats_arr)) * np.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _ensure_businesses():
    global _businesses
    if _businesses is None:
        _businesses = _load_businesses(
            os.path.join(BASE_DIR, "Business_Entities_in_Colorado_Geocoded.csv")
        )


def get_business_at(lat, lon):
    _ensure_businesses()
    dist_km = _haversine_km(lat, lon, _businesses["lats"], _businesses["lons"])
    idx = int(np.argmin(dist_km))
    return _businesses["names"][idx]


def _count_businesses_within(lat, lon, radius_km=RADIUS_KM):
    """Count businesses within radius_km of (lat, lon)."""
    _ensure_businesses()
    dist_km = _haversine_km(lat, lon, _businesses["lats"], _businesses["lons"])
    return int(np.sum(dist_km <= radius_km))


def _nlcd_mode_within(lat, lon, radius_ft=RADIUS_FT):
    """Return the modal NLCD class within radius_ft feet as a one-hot vector."""
    header, data = _nlcd_data
    proj_x, proj_y = _to_albers.transform(lon, lat)

    cellsize = header["cellsize"]
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    nodata = header["nodata_value"]
    radius_cells = int(np.ceil((radius_ft * 0.3048) / cellsize))

    center_col = int((proj_x - xll) / cellsize)
    center_row = int((yll + nrows * cellsize - proj_y) / cellsize)

    values = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr ** 2 + dc ** 2 > radius_cells ** 2:
                continue
            r, c = center_row + dr, center_col + dc
            if 0 <= r < nrows and 0 <= c < ncols:
                val = int(data[r][c])
                if val != nodata:
                    values.append(val)

    if not values:
        return [0] * len(NLCD_CLASSES)
    mode_class = max(set(values), key=values.count)
    return [1 if cls == mode_class else 0 for cls in NLCD_CLASSES]


def _load_ordinances(filepath):
    ordinances = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Name"]
            ordinances[name] = [
                int(row["Fully_Shielded"]),
                int(row["Max_Limit"]),
                int(row["Floodlights"]),
                int(row["Height_Restrictions"]),
            ]
    return ordinances

def get_dark_sky_data(lat, lon):
    #Fetch the data at the lat long
    viirs_val = _grid_lookup(_viirs_data[0], _viirs_data[1], lat, lon)

    proj_x, proj_y = _to_albers.transform(lon, lat)
    nlcd_val = _grid_lookup(_nlcd_data[0], _nlcd_data[1], proj_y, proj_x)

    county_name = _find_county(_county_data[0], _county_data[1], lat, lon)

    ordinance_vec = None
    if county_name and county_name in _ordinances:
        ordinance_vec = _ordinances[county_name]

    business = get_business_at(lat, lon)

    return {
        "viirs": float(viirs_val) if viirs_val is not None else None,
        "nlcd": int(nlcd_val) if nlcd_val is not None else None,
        "county": county_name,
        "business": business,
        "ordinances": ordinance_vec,
    }

def build_regression_matrix(sample_points):
    """Build feature matrix X and target vector y for linear regression.

    For each (lat, lon) in sample_points the feature row contains:
      - 15 NLCD one-hot features (modal land-cover class within 250 ft)
      - 1  business count within 250 ft
      - 4  ordinance flags (Fully_Shielded, Max_Limit, Floodlights, Height_Restrictions)
    Target y is the VIIRS radiance value at that point.
    Points where VIIRS data is missing are skipped.
    """
    X_rows, y_vals = [], []
    for i, (lat, lon) in enumerate(sample_points):
        viirs_val = _grid_lookup(_viirs_data[0], _viirs_data[1], lat, lon)
        if viirs_val is None:
            continue

        nlcd_onehot = _nlcd_mode_within(lat, lon)
        # biz_count = _count_businesses_within(lat, lon)

        # county_name = _find_county(_county_data[0], _county_data[1], lat, lon)
        # if county_name and county_name in _ordinances:
        #     ordinance_vec = _ordinances[county_name]
        # else:
        #     ordinance_vec = [0, 0, 0, 0]

        # X_rows.append(nlcd_onehot + [biz_count] + ordinance_vec)
        X_rows.append(nlcd_onehot)
        y_vals.append(float(viirs_val))

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(sample_points)} points ({len(y_vals)} valid)",
                  flush=True)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y


def linearRegression(sample_points):
    """Fit and report a linear regression model over the given sample points."""
    print(f"Building matrix for {len(sample_points)} sample points...", flush=True)
    X, y = build_regression_matrix(sample_points)
    print(f"Matrix shape: X={X.shape}, y={y.shape}", flush=True)

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    print("\nLinear Regression Coefficients:")
    for name, coef in zip(FEATURE_NAMES, reg.coef_):
        print(f"  {name}: {coef:.6f}")
    print(f"  intercept:  {reg.intercept_:.6f}")
    print(f"  R² score:   {reg.score(X, y):.4f}")
    return reg, X, y




def _geocode_address(address):
    import urllib.request
    import urllib.parse
    import json
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
        "q": address,
        "format": "json",
        "limit": 1,
    })
    req = urllib.request.Request(url, headers={"User-Agent": "dark-sky-research/1.0"})
    with urllib.request.urlopen(req) as resp:
        results = json.loads(resp.read())
    if not results:
        return None, None
    return float(results[0]["lat"]), float(results[0]["lon"])


def _preload_data():
    global _viirs_data, _nlcd_data, _county_data, _ordinances
    print("Loading raster files...", flush=True)
    t0 = time.perf_counter()
    _viirs_data = _parse_asc(os.path.join(BASE_DIR, "colorado_2023_viirs.asc"))
    _nlcd_data = _parse_asc(os.path.join(BASE_DIR, "ncld_2016 1 (1).asc"), dtype=np.int16)
    _county_data = _load_county_data(os.path.join(BASE_DIR, "Counties_colorado_2005"))
    _ordinances = _load_ordinances(os.path.join(BASE_DIR, "colorado_dark_sky_ordinances.csv"))
    print(f"Done ({time.perf_counter() - t0:.2f}s)\n", flush=True)

# bolder
# 40.025166801818 (Lat)
# -105.242743364816 (Long)

# El Paso
# 38.947795585574
#-104.87280185668

if __name__ == "__main__":
    _preload_data()

    mode = input("Mode — (1) lookup single point  (2) run linear regression: ").strip()

    if mode == "1":
        lat = float(input("Enter Lat: ").strip())
        lon = float(input("Enter Long: ").strip())
        result = get_dark_sky_data(lat, lon)
        print(result)

    elif mode == "2":
        # Sample a regular grid over Colorado's bounding box.
        # Step ~0.1° ≈ 7 miles; gives ~2,400 candidate points before VIIRS filtering.
        step = float(input("Grid step in degrees (e.g. 0.1): ").strip() or "0.1")
        lats = np.arange(37.0, 41.01, step)
        lons = np.arange(-109.05, -102.04, step)
        sample_points = [(la, lo) for la in lats for lo in lons]
        print(f"Generated {len(sample_points)} candidate grid points.")
        linearRegression(sample_points)

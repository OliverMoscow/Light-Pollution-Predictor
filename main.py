import csv
import os
import time
import numpy as np
import shapefile
import matplotlib.pyplot as plt
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

def _precompute_nlcd_fracs():
    """Precompute the fraction of each NLCD class within RADIUS_FT for every cell.
    Returns float16 array of shape (15, nrows, ncols) where each value is the
    proportion of that class within the circular neighborhood (sums to 1 per cell)."""
    from scipy.ndimage import convolve
    header, data = _nlcd_data
    cellsize = header["cellsize"]
    radius_cells = int(np.ceil((RADIUS_FT * 0.3048) / cellsize))

    # Circular footprint
    d = np.arange(-radius_cells, radius_cells + 1)
    xx, yy = np.meshgrid(d, d)
    footprint = (xx ** 2 + yy ** 2 <= radius_cells ** 2).astype(np.float32)

    # One convolution per class → count of that class within radius at every cell
    counts = np.zeros((len(NLCD_CLASSES), data.shape[0], data.shape[1]), dtype=np.float32)
    for k, cls in enumerate(NLCD_CLASSES):
        binary = (data == cls).astype(np.float32)
        counts[k] = convolve(binary, footprint, mode="constant", cval=0)

    total = counts.sum(axis=0)
    # Divide each class count by total to get proportion; 0 where no data
    fracs = np.where(total > 0, counts / np.maximum(total, 1), 0).astype(np.float16)
    return fracs


def _precompute_county_grid():
    """Rasterize county assignments onto the VIIRS grid.
    Returns an int16 array (nrows, ncols) with the county index, or -1."""
    from matplotlib.path import Path
    header, _ = _viirs_data
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    cellsize = header["cellsize"]

    lats_vec = yll + (nrows - np.arange(nrows) - 0.5) * cellsize
    lons_vec = xll + (np.arange(ncols) + 0.5) * cellsize
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)
    pts = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])  # (N, 2)

    shapes, _ = _county_data
    county_grid = np.full(nrows * ncols, -1, dtype=np.int16)
    for i, s in enumerate(shapes):
        bb = s.bbox
        bbox_mask = ((pts[:, 0] >= bb[0]) & (pts[:, 0] <= bb[2]) &
                     (pts[:, 1] >= bb[1]) & (pts[:, 1] <= bb[3]))
        if not bbox_mask.any():
            continue
        inside = np.zeros(len(pts), dtype=bool)
        inside[bbox_mask] = Path(s.points).contains_points(pts[bbox_mask])
        county_grid[inside] = i

    return county_grid.reshape(nrows, ncols)


def build_regression_matrix(nlcd_fracs, county_grid):
    """Build feature matrix X and target vector y using precomputed rasters.
    All lookups are vectorized numpy array indexing — no Python loops over points.

    Features per point:
      - 15 NLCD fraction features (proportion of each class within 250 ft)
      - 4  ordinance flags (Fully_Shielded, Max_Limit, Floodlights, Height_Restrictions)
    Target y is the VIIRS radiance value.
    """
    v_header, v_data = _viirs_data
    n_header, _ = _nlcd_data
    _, county_names = _county_data

    nrows_v = int(v_header["nrows"])
    xll_v, yll_v = v_header["xllcorner"], v_header["yllcorner"]
    cs_v = v_header["cellsize"]
    nodata_v = v_header["nodata_value"]

    xll_n, yll_n = n_header["xllcorner"], n_header["yllcorner"]
    cs_n = n_header["cellsize"]
    nrows_n, ncols_n = int(n_header["nrows"]), int(n_header["ncols"])

    # All valid VIIRS cells
    v_rows, v_cols = np.where(v_data != nodata_v)
    y = v_data[v_rows, v_cols].astype(np.float32)

    # Cell-center coordinates
    lats = yll_v + (nrows_v - v_rows - 0.5) * cs_v
    lons = xll_v + (v_cols + 0.5) * cs_v

    # NLCD: batch-transform to Albers → index into precomputed fractions raster
    proj_xs, proj_ys = _to_albers.transform(lons, lats)
    n_cols = np.clip(((proj_xs - xll_n) / cs_n).astype(int), 0, ncols_n - 1)
    n_rows = np.clip(((yll_n + nrows_n * cs_n - proj_ys) / cs_n).astype(int), 0, nrows_n - 1)
    # nlcd_fracs shape: (15, nrows, ncols) → index to get (15, n_pts) → transpose to (n_pts, 15)
    nlcd_matrix = nlcd_fracs[:, n_rows, n_cols].T.astype(np.float32)

    # Ordinances: county index → ordinance vector
    c_idx = county_grid[v_rows, v_cols]
    ordinance_matrix = np.zeros((len(y), 4), dtype=np.float32)
    for ci, name in enumerate(county_names):
        if name in _ordinances:
            ordinance_matrix[c_idx == ci] = _ordinances[name]

    X = np.hstack([nlcd_matrix, ordinance_matrix])
    return X, y


def linearRegression():
    """Precompute rasters, build the matrix, fit and report the linear regression."""
    print("Precomputing NLCD fraction raster (15 convolutions)...", flush=True)
    t0 = time.perf_counter()
    nlcd_fracs = _precompute_nlcd_fracs()
    print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Rasterizing counties onto VIIRS grid...", flush=True)
    t0 = time.perf_counter()
    county_grid = _precompute_county_grid()
    print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Building regression matrix...", flush=True)
    t0 = time.perf_counter()
    X, y = build_regression_matrix(nlcd_fracs, county_grid)
    print(f"  done ({time.perf_counter() - t0:.1f}s)  shape: X={X.shape}, y={y.shape}", flush=True)

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    print("\nLinear Regression Coefficients:")
    for name, coef in zip(FEATURE_NAMES, reg.coef_):
        print(f"  {name}: {coef:.6f}")
    print(f"  intercept:  {reg.intercept_:.6f}")
    print(f"  R² score:   {reg.score(X, y):.4f}")

    y_pred = reg.predict(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, y_pred, alpha=0.4, s=10, label="samples")
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect fit")
    ax.set_xlabel("Actual VIIRS radiance")
    ax.set_ylabel("Predicted VIIRS radiance")
    ax.set_title(f"Predicted vs Actual VIIRS  (R²={reg.score(X, y):.4f})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return reg, X, y

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
        linearRegression()

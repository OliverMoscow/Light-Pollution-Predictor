import csv
import os
import time
import numpy as np
import shapefile
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial import KDTree
from matplotlib.path import Path
from sklearn import linear_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RADIUS_KM = 0.5  # search radius for nearby businesses

BUSINESSES_FILE = "osm_businesses_colorado.csv"


def parse_asc(filepath, dtype=np.float32):
    with open(filepath) as f:
        header = {}
        for _ in range(6):
            parts = f.readline().split()
            header[parts[0].lower()] = float(parts[1])
    data = np.loadtxt(filepath, skiprows=6, dtype=dtype)
    return header, data


def load_businesses(filepath):
    lats, lons = [], []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
            except (ValueError, KeyError):
                pass
    return np.array(lats), np.array(lons)


def load_ordinances(filepath):
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
    sf = shapefile.Reader(filepath)
    shapes = sf.shapes()
    names = [rec["NAME"] for rec in sf.iterRecords()]
    return shapes, names


def rasterize_county_grid(shapes, names, header):
    """Assign a county index to every VIIRS pixel center."""
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    xll, yll, cs = header["xllcorner"], header["yllcorner"], header["cellsize"]

    lats_vec = yll + (nrows - np.arange(nrows) - 0.5) * cs
    lons_vec = xll + (np.arange(ncols) + 0.5) * cs
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)
    pts = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])

    county_grid = np.full(nrows * ncols, -1, dtype=np.int16)
    for i, s in enumerate(shapes):
        bb = s.bbox
        mask = ((pts[:, 0] >= bb[0]) & (pts[:, 0] <= bb[2]) &
                (pts[:, 1] >= bb[1]) & (pts[:, 1] <= bb[3]))
        if not mask.any():
            continue
        inside = np.zeros(len(pts), dtype=bool)
        inside[mask] = Path(s.points).contains_points(pts[mask])
        county_grid[inside] = i

    return county_grid.reshape(nrows, ncols)


def build_business_score_grid(biz_lats, biz_lons, header):
    """For each pixel, sum 1/sqrt(dist_km) for all businesses within RADIUS_KM.
    Closer businesses contribute more (inverse square-root weighting)."""
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    xll, yll, cs = header["xllcorner"], header["yllcorner"], header["cellsize"]

    lats_vec = yll + (nrows - np.arange(nrows) - 0.5) * cs
    lons_vec = xll + (np.arange(ncols) + 0.5) * cs
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)

    pixel_lats = lat_mesh.ravel()
    pixel_lons = lon_mesh.ravel()

    # KD-tree in lat/lon degrees; convert radius to degrees (1 deg ≈ 111 km)
    tree = KDTree(np.column_stack([biz_lats, biz_lons]))
    pixel_pts = np.column_stack([pixel_lats, pixel_lons])
    neighbors_list = tree.query_ball_point(pixel_pts, r=RADIUS_KM / 111.0)

    scores = np.zeros(len(pixel_pts), dtype=np.float32)
    for i, neighbors in enumerate(neighbors_list):
        if not neighbors:
            continue
        dlat = (biz_lats[neighbors] - pixel_lats[i]) * 111.0
        dlon = (biz_lons[neighbors] - pixel_lons[i]) * 111.0 * np.cos(np.radians(pixel_lats[i]))
        dist_km = np.maximum(np.sqrt(dlat**2 + dlon**2), 0.01)
        scores[i] = np.sum(1.0 / np.sqrt(dist_km))

    return scores.reshape(nrows, ncols)


def build_regression_matrix(viirs_header, viirs_data, biz_scores, county_grid, county_names, ordinances):
    """Build feature matrix X (business score + 4 ordinance flags) and target y."""
    nodata = viirs_header["nodata_value"]
    rows, cols = np.where(viirs_data != nodata)
    y_all = viirs_data[rows, cols].astype(np.float32)

    X_rows, valid = [], []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        county_idx = county_grid[r, c]
        if county_idx < 0:
            continue
        county_name = county_names[county_idx]
        if county_name not in ordinances:
            continue
        X_rows.append([biz_scores[r, c]] + ordinances[county_name])
        valid.append(idx)

    return np.array(X_rows, dtype=np.float32), y_all[valid]


def run_regression(label, biz_lats, biz_lons, viirs_header, viirs_data, county_grid, county_names, ordinances):
    """Fit and report a linear regression for one business data source. Returns (reg, X, y)."""
    print(f"\n[{label}] Building business score grid...", flush=True)
    t0 = time.perf_counter()
    biz_scores = build_business_score_grid(biz_lats, biz_lons, viirs_header)
    print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True)

    X, y = build_regression_matrix(viirs_header, viirs_data, biz_scores, county_grid, county_names, ordinances)
    print(f"  X shape: {X.shape}, y shape: {y.shape}", flush=True)

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    feature_names = ["biz_score", "fully_shielded", "max_limit", "floodlights", "height_restrictions"]
    print(f"  Coefficients:")
    for name, coef in zip(feature_names, reg.coef_):
        print(f"    {name}: {coef:.6f}")
    print(f"  intercept: {reg.intercept_:.6f}")
    print(f"  R²:        {reg.score(X, y):.4f}")

    return reg, X, y


def main():
    print("Loading VIIRS...", flush=True)
    viirs_header, viirs_data = parse_asc(os.path.join(BASE_DIR, "colorado_2023_viirs.asc"))

    print("Loading counties...", flush=True)
    shapes, county_names = load_county_data(os.path.join(BASE_DIR, "Counties_colorado_2005"))

    print("Loading ordinances...", flush=True)
    ordinances = load_ordinances(os.path.join(BASE_DIR, "colorado_dark_sky_ordinances.csv"))

    print("Rasterizing counties...", flush=True)
    t0 = time.perf_counter()
    county_grid = rasterize_county_grid(shapes, county_names, viirs_header)
    print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Loading OSM businesses...", flush=True)
    biz_lats, biz_lons = load_businesses(os.path.join(BASE_DIR, BUSINESSES_FILE))
    print(f"  {len(biz_lats):,} businesses loaded", flush=True)

    reg, X, y = run_regression("osm", biz_lats, biz_lons, viirs_header, viirs_data,
                               county_grid, county_names, ordinances)

    y_pred = reg.predict(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, y_pred, alpha=0.3, s=5)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual VIIRS radiance")
    ax.set_ylabel("Predicted VIIRS radiance")
    ax.set_title(f"Predicted vs Actual  (R²={reg.score(X, y):.4f})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import joblib
from fetch_osm_colorado import fetch_osm_businesses, CACHE_CSV, load_vocab, decode_tag
import time
import shapefile as sf_module
from shapely.geometry import shape
from shapely import STRtree, points as shp_points




N_TAGS = 1000  # number of top tags (by frequency) to keep

def loadVIIRSData(): 
    #read asc file
    filepath = "colorado_2023_viirs.asc"
    header = {}
    header_rows = 6  # standard .asc header is 6 lines

    # --- Parse header ---
    with open(filepath, 'r') as f:
        for _ in range(header_rows):
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1])

    ncols     = int(header['ncols'])
    nrows     = int(header['nrows'])
    xll       = header['xllcorner'] # left edge
    yll       = header['yllcorner'] # bottom edge
    cellsize  = header['cellsize']
    nodata    = header['nodata_value']

    # --- Parse raster values ---
    # skiprows skips the 6 header lines
    flat_values = np.loadtxt(filepath, skiprows=header_rows).ravel()

    # --- Build coordinate arrays ---
    # X (lon): left edge + half cell to center, across all columns
    # Y (lat): bottom edge + (nrows - row - 0.5) * cellsize (top-down raster)
    cols = np.arange(ncols)
    rows = np.arange(nrows)

    lons = xll + (cols + 0.5) * cellsize          # shape: (ncols,)
    lats = yll + (nrows - rows - 0.5) * cellsize  # shape: (nrows,)

    # Broadcast to full grid
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # both shape: (nrows, ncols)

    # --- Build DataFrame ---
    pixels = np.column_stack([
        lon_grid.ravel(),
        lat_grid.ravel(),
        flat_values
    ])

    # Remove nodata rows
    pixels = pixels[pixels[:, 2] != nodata]

    #Get bounds of raster
    xmin = xll
    ymin = yll
    xmax = xll + (ncols * cellsize)
    ymax = yll + (nrows * cellsize)

    bounds = {
        'bottom_left':  (xmin, ymin),
        'bottom_right': (xmax, ymin),
        'top_left':     (xmin, ymax),
        'top_right':    (xmax, ymax),
    }

    #create 2d grid
    grid = flat_values.reshape(nrows, ncols).copy()
    grid[grid == nodata] = np.nan

    return pixels, bounds, grid, cellsize

def loadOSMData(bounds):
    # Fetch data from overpass api.
    xmin, ymin = bounds['bottom_left']
    xmax, ymax = bounds['top_right']
    bbox = f"{ymin},{xmin},{ymax},{xmax}"

    fetch_osm_businesses(bbox)  # no-op if bounds unchanged, else fetches + saves CSV

    df = pd.read_csv(CACHE_CSV)
    df = df[df["tag_idx"] < N_TAGS].reset_index(drop=True)
    return df[["lat", "lon", "tag_idx", "tag_count"]].to_numpy(dtype=np.float32)

def exploreHeatMap(grid):
    plt.figure(figsize=(12, 8))
    plt.imshow((grid), origin='upper', cmap='inferno', aspect='auto')
    plt.colorbar(label='log(radiance)')
    plt.show()


def inspectOutliers(viirs, grid, threshold=5.0):
    mask = viirs[:, 2] > threshold
    outliers = viirs[mask]
    print(f"total pixels: {len(viirs)}")
    print(f"=== Outlier Pixels (radiance > {threshold}) ===")
    print(f"  count : {len(outliers)}")
    print(f"  coords span lon {outliers[:,0].min():.3f} → {outliers[:,0].max():.3f}")
    print(f"  coords span lat {outliers[:,1].min():.3f} → {outliers[:,1].max():.3f}")


def inspectRadiancePerGroup(osm_with_radiance):
    """
    osm_with_radiance: (M, 5) — [lat, lon, tag_idx, tag_count, viirs_radiance]
    """
    vocab         = load_vocab()
    groups        = osm_with_radiance[:, 2].astype(int)
    radiance      = osm_with_radiance[:, 4]
    unique_groups = np.unique(groups)
    group_names   = [decode_tag(g, vocab) for g in unique_groups]

    # --- Box plot ---
    # How to read: each box shows the middle 50% of radiance values for that tag
    # (25th–75th percentile). The line inside is the median. Whiskers extend to
    # the 5nd and 95th percentiles. Outliers are hidden to keep the chart readable.
    # A box sitting higher on the y-axis = that tag tends to appear near brighter pixels.

    cmap       = plt.get_cmap("tab20")
    tag_colors = {name: cmap(i % 20) for i, name in enumerate(group_names)}

    df_box = pd.DataFrame({"tag": [group_names[list(unique_groups).index(g)] for g in groups], "radiance": radiance})
    tag_counts = df_box["tag"].value_counts()
    df_box = df_box[df_box["tag"].isin(tag_counts[tag_counts >= 1000].index)]
    order  = df_box.groupby("tag")["radiance"].median().sort_values().index.tolist()
    tag_colors = {name: cmap(i % 20) for i, name in enumerate(order)}

    fig1, ax1 = plt.subplots(figsize=(18, 8))
    sns.boxplot(data=df_box, x="tag", y="radiance", order=order,
                palette={name: tag_colors[name] for name in order},
                whis=[5, 95], flierprops={"marker": ""}, ax=ax1)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    for tick in ax1.get_xticklabels():
        tick.set_color(tag_colors[tick.get_text()])
    ax1.set_ylabel('radiance')
    ax1.set_title('Radiance distribution per tag')
    fig1.tight_layout()
    plt.show()

    # --- Top 10 tags per radiance bin (% of each tag's own total in this band) ---
    # n_bins    = 8
    # bin_edges = np.linspace(0, radiance.max(), n_bins + 1)

    # # Total count per tag across all bins
    # tag_totals = {g: np.sum(groups == g) for g in unique_groups}

    # # Consistent color per tag name
    # cmap       = plt.get_cmap("tab20")
    # tag_colors = {name: cmap(i % 20) for i, name in enumerate(group_names)}

    # ncols = 4
    # nrows = (n_bins + ncols - 1) // ncols
    # fig2, axes2 = plt.subplots(nrows, ncols, figsize=(22, nrows * 5))
    # axes2 = axes2.ravel()

    # for i in range(n_bins):
    #     lo, hi = bin_edges[i], bin_edges[i + 1]
    #     mask       = (radiance >= lo) & (radiance <= hi)
    #     bin_groups = groups[mask]
    #     if len(bin_groups) == 0:
    #         axes2[i].set_visible(False)
    #         continue

    #     unique_b, counts_b = np.unique(bin_groups, return_counts=True)
    #     # % of each tag's own total that lands in this bin
    #     pct       = np.array([counts_b[j] / tag_totals[unique_b[j]] * 100 for j in range(len(unique_b))])
    #     top       = np.argsort(pct)[-20:]
    #     top_names = [decode_tag(unique_b[j], vocab) for j in top]
    #     top_pct   = pct[top]
    #     colors    = [tag_colors[name] for name in top_names]

    #     axes2[i].barh(top_names, top_pct, color=colors)
    #     axes2[i].tick_params(axis='y', labelsize=7)
    #     axes2[i].set_xlabel('% of tag\'s total count', fontsize=8)
    #     axes2[i].set_title(f'radiance {lo:.1f} – {hi:.1f}', fontsize=9)

    # for j in range(n_bins, len(axes2)):
    #     axes2[j].set_visible(False)

    # fig2.suptitle('Top 10 tags per radiance band  (% of each tag\'s own total)', fontsize=14)
    # fig2.tight_layout()
    # plt.show()

def getNearBusinesses(viirs, osm, cellsize, radius=0.05, n_tags=N_TAGS):
    """
    For each VIIRS pixel, sum inverse-square-distance weights of nearby businesses
    grouped by category.

    Returns
    -------
    X : (N, n_tags) float32 — weighted business density per category per pixel
    y : (N,)        float32 — VIIRS radiance per pixel

    Example input / output
    X = [0.1, 0.3, 0, 0, 3] where each index represents a category and is the linear
    interpolation of nearby businesses of a viirs pixel
    y = [22.2] a viirs reading at a certain pixel
    """

    viirs_coords = viirs[:, 0:2]          # (N, 2)  lon, lat
    viirs_values = viirs[:, 2]            # (N,)    radiance

    osm_coords = osm[:, 1::-1]            # (M, 2)  flip lat,lon → lon,lat to match viirs
    osm_tags   = osm[:, 2].astype(int)    # (M,)    tag index

    #convert cell size from degrees to km
    meters_per_degree = 111_000 * np.cos(np.radians(39))
    cellsize_meters = cellsize * meters_per_degree  # ~250m
    cellsize_km = cellsize_meters / 1000            # ~0.25km
    cell_area_km2 = cellsize_km ** 2               # ~0.0625 km²

    # Build tree on viirs pixels; for each business find all pixels within radius
    tree = cKDTree(viirs_coords)
    neighbor_lists = tree.query_ball_point(osm_coords, r=radius, workers=-1)

    N = len(viirs)
    X = np.zeros((N, n_tags), dtype=np.float32)

    M = len(neighbor_lists)
    t0 = time.time()
    print(f"getNearBusinesses: processing {M:,} businesses...")
    for biz_idx, pixel_indices in enumerate(neighbor_lists):
        if not pixel_indices:
            continue
        tag = osm_tags[biz_idx]
        if tag < 0 or tag >= n_tags:
            continue
        pixel_indices = np.asarray(pixel_indices)
        dists = np.linalg.norm(viirs_coords[pixel_indices] - osm_coords[biz_idx], axis=1)
        dists = np.maximum(dists, 1e-6)          # avoid div-by-zero for exact matches
        #adding a weight based on inverse distance ^2 
        weights = 1.0 / dists
        #normalizing by cellsize 
        weights /= cell_area_km2
        X[pixel_indices, tag] += weights
    print(f"getNearBusinesses: done in {time.time() - t0:.1f}s")

    #adding log to smooth the jumps in cell size influence
    X = np.log1p(X)

    return X, viirs_values.astype(np.float32)

def businessCorrelation(viirs, osm):
    # Build a KD-tree on pixel coords
    viirs_coords = viirs[:, 0:2]          # (N, 2) lon, lat
    viirs_values = viirs[:, 2]            # (N,)   radiance

    osm_coords = osm[:, 1::-1]           # (M, 2) flip lat,lon → lon,lat to match viirs

    # Build tree on viirs pixel coords, query with business coords
    tree = cKDTree(viirs_coords)
    _, nearest_idx = tree.query(osm_coords, workers=-1)   # workers=-1 uses all CPU cores

    return viirs_values[nearest_idx]


def loadCountyOrdinances(shp_path="Counties_colorado_2005.shp",
                         ord_path="colorado_dark_sky_ordinances.csv"):
    """
    Load county polygons and dark-sky ordinance data.

    Returns
    -------
    county_geoms  : list of shapely Polygon/MultiPolygon  (len = n_counties)
    county_names  : list of str  (county NAME field, len = n_counties)
    ordinance_df  : DataFrame indexed by county name,
                    columns = ['Fully_Shielded', 'Max_Limit', 'Floodlights', 'Height_Restrictions']
    """
    ordinance_df = pd.read_csv(ord_path).set_index("Name")

    reader = sf_module.Reader(shp_path)
    field_names = [f[0] for f in reader.fields[1:]]
    name_idx = field_names.index("NAME")

    county_geoms = []
    county_names = []
    for i in range(len(reader)):
        rec = reader.record(i)
        county_names.append(rec[name_idx])
        county_geoms.append(shape(reader.shape(i).__geo_interface__))

    return county_geoms, county_names, ordinance_df


def getCountyOrdinanceFeatures(viirs, county_geoms, county_names, ordinance_df):
    """
    For each VIIRS pixel (lon, lat), determine which county it falls in and
    return the ordinance binary features for that county (0 if no ordinance).

    Returns
    -------
    X_ord            : (N, n_ord_features) float32
    ord_feature_names: list of str
    """
    ord_feature_names = ordinance_df.columns.tolist()
    n_ord = len(ord_feature_names)
    N = len(viirs)

    # Build spatial index over county polygons
    tree = STRtree(county_geoms)

    # Create a shapely Point array for all pixels — shapely 2.x vectorised constructor
    pixel_points = shp_points(viirs[:, 0], viirs[:, 1])  # (N,) array of Points

    # query returns (2, k): row0 = pixel indices, row1 = county indices
    pix_idx, cty_idx = tree.query(pixel_points, predicate="within")

    X_ord = np.zeros((N, n_ord), dtype=np.float32)
    for pi, ci in zip(pix_idx, cty_idx):
        cname = county_names[ci]
        if cname in ordinance_df.index:
            X_ord[pi] = ordinance_df.loc[cname].values.astype(np.float32)

    covered = np.count_nonzero(pix_idx)
    print(f"getCountyOrdinanceFeatures: {covered:,} / {N:,} pixels assigned to a county")
    return X_ord, ord_feature_names


if __name__ == "__main__":


    #every random should use the same seed
    np.random.seed(42)
    # Load data to memory
    viirs, bounds, grid, cellsize = loadVIIRSData()
    osm = loadOSMData(bounds)

    #Data analysis
    osm_radiance = businessCorrelation(viirs, osm)   # (M,) — one value per business
    osm_full = np.column_stack([osm, osm_radiance])  # (M, 5)
    inspectRadiancePerGroup(osm_full)

    
    inspectRadiancePerGroup(osm_full)
    inspectOutliers(viirs, grid)

    # # Build linear regression model
    # print(f"Building feature matrix X for {len(viirs):,} pixels and {len(osm):,} businesses...")
    # t0 = time.time()
    # X, y = getNearBusinesses(viirs, osm, cellsize)
    # print(f"Feature matrix built in {time.time() - t0:.1f}s  shape={X.shape}")

    # print(f"Avg businesses per pixel: {(X > 0).sum(axis=1).mean():.2f}")
    # print(f"Max businesses per pixel: {(X > 0).sum(axis=1).max()}")

    # # Keep only pixels that have at least one nearby business. Otherwise, we have a ton
    # # of pixels with no data which makes the model slow to train and overfits to predict
    # # a low value.  
    # has_business = X.sum(axis=1) > 0
    # X, y = X[has_business], y[has_business]
    # print(f"Filtered to {has_business.sum():,} pixels with nearby businesses (dropped {(~has_business).sum():,})")
    
    
    # # 1. Split first — always before any normalization
    # print(f"Splitting {len(X):,} pixels into train/val (80/20)...")
    # t_split = time.time()
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(f"  train={len(X_train):,}  val={len(X_val):,}  ({time.time()-t_split:.1f}s)")

    # #load saved model
    # saved = joblib.load("model.joblib")
    # reg, scaler = saved["model"], saved["scaler"]

    # X_new_scaled = scaler.transform(X_val)
    # y_pred = reg.predict(X_new_scaled)

    # # #2. Fit scaler ONLY on training data
    # # print(f"Scaling features (StandardScaler) on {X_train.shape}...")
    # # t_scale = time.time()
    # # scaler = StandardScaler()
    # # X_train = scaler.fit_transform(X_train)
    # # X_val = scaler.transform(X_val)
    # # print(f"  done in {time.time()-t_scale:.1f}s  mean≈{scaler.mean_.mean():.3f}  std≈{scaler.scale_.mean():.3f}")

    # # # plt.hist(y_train, bins=100, log=True)  # log scale to see the tail
    # # # plt.xlabel("VIIRS radiance")
    # # # plt.ylabel("Count (log scale)")
    # # # plt.show()

    # # print("Fitting linear regression...")
    # # t1 = time.time()
    
    # # reg = linear_model.LinearRegression()
    # # reg.fit(X_train, y_train)
    # # print(f"Training done in {time.time() - t1:.1f}s")
    # # joblib.dump({"model": reg, "scaler": scaler}, "model.joblib")
    # # print("Model saved to model.joblib")
    # coef = reg.coef_
    # intercept = reg.intercept_

    # print("coeficient shape: ", coef.shape)
    # print("Intercept: ", intercept)
    # print("check 1: Verify loss at initialization")
    # print("real loss: ", y_train.var())
    # print("Predicting on training set...")
    # y_pred = reg.predict(X_train)
    # mse = mean_squared_error(y_train, y_pred)
    # print(f"Train MSE: {mse:.4f}  (baseline variance: {y_train.var():.4f})  R²: {1 - mse/y_train.var():.4f}")
    # y_pred_val = reg.predict(X_val)
    # mse_val = mean_squared_error(y_val, y_pred_val)
    # print(f"Val   MSE: {mse_val:.4f}  R²: {1 - mse_val/y_val.var():.4f}")

    # print("check 2: input-independent baseline")
    # X_zeros = np.zeros_like(X_train)
    # y_pred_zeros = reg.predict(X_zeros)
    # mse_zeros = mean_squared_error(y_train, y_pred_zeros)
    # print(f"Zero-input MSE: {mse_zeros:.4f}")

    # print("check 3: overfit on a few samples. MSE should go to near zero")
    # # grab just 10 examples
    # X_tiny = X_train[:10]
    # y_tiny = y_train[:10]

    # # fit on just those
    # reg.fit(X_tiny, y_tiny)
    # y_pred_tiny = reg.predict(X_tiny)
    # mse_tiny = mean_squared_error(y_tiny, y_pred_tiny)
    # print(f"Tiny batch MSE: {mse_tiny:.4f}")

    # print("check 4: inspect the feature pipeline visually")
    # vocab = load_vocab()

    # for i in range(5):
    #     top_idxs = np.argsort(X_train[i])[-5:][::-1]
    #     top = [(decode_tag(j, vocab), round(float(X_train[i][j]), 4)) for j in top_idxs]
    #     print(f"Pixel {i} | VIIRS: {y_train[i]:.2f} | Top categories: {top}")
    

    # print("Model Mistake Analysis")
    # y_pred_val = reg.predict(X_val)
    # residuals = y_val - y_pred_val

    # # Basic error stats
    # print(f"Mean error: {residuals.mean():.4f}")
    # print(f"Std of errors: {residuals.std():.4f}")
    # print(f"Max overpredict: {residuals.min():.4f}")
    # print(f"Max underpredict: {residuals.max():.4f}")


    # # 1. Replace scatter with 2D density plot
    # # try:
    # #     sns.displot(x=y_val, y=y_pred_val, kind="kde", fill=True)
    # # except ValueError:
    # #     fig, ax = plt.subplots()
    # #     ax.hexbin(y_val, y_pred_val, gridsize=50, cmap="Blues")
    # # plt.xlabel("True VIIRS radiance")
    # # plt.ylabel("Predicted radiance")
    # # plt.title("True vs predicted density")
    # # plt.show()

    # # 2. Residual histogram colored by a categorical attribute
    # # First you need something to color by — e.g. dominant category per pixel
    # dominant_tag = np.argmax(X_val, axis=1)
    # dominant_name = [decode_tag(i, vocab) for i in dominant_tag]

    # df_resid = pd.DataFrame({
    #     "residual": residuals,
    #     "dominant_tag": dominant_name,
    #     "true_radiance_bin": pd.cut(y_val, bins=5)  # or try coloring by radiance bucket
    # })

    # all_tags = df_resid["dominant_tag"].value_counts().index
    # tag_stats = []
    # for tag in all_tags:
    #     mask = df_resid["dominant_tag"] == tag
    #     tag_stats.append((tag, mask.sum(), df_resid[mask]["residual"].mean()))
    # tag_stats.sort(key=lambda x: x[2])

    # print("\nResiduals by dominant tag:")
    # print(f"{'tag':<45} {'count':>6}  {'mean_residual':>13}  {'interpretation'}")
    # print("-" * 90)
    # for tag, count, mean_resid in tag_stats:
    #     if mean_resid > 0.5:
    #         interp = "underpredicting (too dim)"
    #     elif mean_resid < -0.5:
    #         interp = "overpredicting (too bright)"
    #     else:
    #         interp = "accurate"
    #     print(f"{tag:<45} {count:>6}  {mean_resid:>+13.3f}  {interp}")

    # top8_tags = df_resid["dominant_tag"].value_counts().head(8).index
    # df_plot = df_resid[df_resid["dominant_tag"].isin(top8_tags)]
    # df_plot_clipped = df_plot[df_plot["residual"].abs() < 10]
    # sns.histplot(data=df_plot_clipped, x="residual", hue="dominant_tag", bins=50, kde=True)
    # plt.axvline(0, color='red', linewidth=1)
    # plt.title("Residuals by dominant business category")
    # plt.show()

    # vocab = load_vocab()

    # # pair each coefficient with its category name
    # coef_named = [(decode_tag(i, vocab), reg.coef_[i]) for i in range(N_TAGS)]
    # coef_named.sort(key=lambda x: x[1], reverse=True)

    # print("Top 10 brightest predictors (positive coefficients):")
    # for name, coef in coef_named[:10]:
    #     print(f"  {coef:+.4f}  {name}")

    # print("\nTop 10 darkest predictors (negative coefficients):")
    # for name, coef in coef_named[-10:]:
    #     print(f"  {coef:+.4f}  {name}")

    # top_n = 20
    # top = coef_named[:top_n]
    # bottom = coef_named[-top_n:]
    # combined = bottom + top

    # names = [x[0] for x in combined]
    # values = [x[1] for x in combined]
    # colors = ['#d9534f' if v < 0 else '#5cb85c' for v in values]

    # plt.figure(figsize=(10, 10))
    # plt.barh(names, values, color=colors)
    # plt.axvline(0, color='black', linewidth=0.8)
    # plt.xlabel("Coefficient value")
    # plt.title("What the model learned — red=darker, green=brighter")
    # plt.tight_layout()
    # plt.show()

    #Explore data
    # exploreHeatMap(grid)
    # inspectOSM(CACHE_CSV)
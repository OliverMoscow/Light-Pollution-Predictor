import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np

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

    return pixels, bounds, grid

def loadOSMData(bounds):
    # Fetch data from overpass api.
    from fetch_osm_colorado import fetch_osm_businesses, CACHE_CSV
    xmin, ymin = bounds['bottom_left']
    xmax, ymax = bounds['top_right']
    bbox = f"{ymin},{xmin},{ymax},{xmax}"

    fetch_osm_businesses(bbox)  # use stored csv if bounds unchanged, else fetches + saves CSV

    # lat, lon as float32; lightgroup_idx loaded as float then cast to int
    arr = np.loadtxt(CACHE_CSV, delimiter=",", skiprows=1, dtype=np.float32)
    arr[:, 2] = arr[:, 2].astype(np.int32)
    return arr

def exploreHeatMap(grid):
    plt.figure(figsize=(12, 8))
    plt.imshow(np.log1p(grid), origin='upper', cmap='inferno', aspect='auto')
    plt.colorbar(label='log(radiance)')
    plt.show()

def businessCorrelation(viirs, osm):
    # Build a KD-tree on pixel coords
    viirs_coords = viirs[:, 0:2]          # (N, 2) lon, lat
    viirs_values = viirs[:, 2]            # (N,)   radiance

    osm_coords = osm[:, 1::-1]           # (M, 2) flip lat,lon → lon,lat to match viirs

    # Build tree on viirs pixel coords, query with business coords
    tree = cKDTree(viirs_coords)
    _, nearest_idx = tree.query(osm_coords, workers=-1)   # workers=-1 uses all CPU cores

    return viirs_values[nearest_idx]   

if __name__ == "__main__":
    # Load data to memory
    viirs, bounds, grid = loadVIIRSData()
    osm = loadOSMData(bounds)
    exploreHeatMap(grid)
    # businessCorrelation(viirs,osm)

    # Construct business perameter matrix
    # train
import csv
import os
import time
import numpy as np
import shapefile
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_viirs_data = None
_nlcd_data = None
_county_data = None
_ordinances = None
_businesses = None  # dict: "lats", "lons" (np arrays), "names" (list)
# Converts lat/lon (EPSG:4326) to Albers Equal Area meters (EPSG:5070),
# which is the projected coordinate system the NLCD raster uses.
_to_albers = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


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


def get_business_at(lat, lon):
    global _businesses
    if _businesses is None:
        _businesses = _load_businesses(
            os.path.join(BASE_DIR, "Business_Entities_in_Colorado_Geocoded.csv")
        )
    dlat = np.radians(_businesses["lats"] - lat)
    dlon = np.radians(_businesses["lons"] - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(_businesses["lats"])) * np.sin(dlon / 2) ** 2
    dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
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


if __name__ == "__main__":
    _preload_data()
    lat = float(input("Enter Lat: ").strip())
    lon = float(input("Enter Long: ").strip())
    result = get_dark_sky_data(lat, lon)
    print(result)

"""
Geocodes Colorado business entity addresses using the US Census Bureau batch geocoder.
Replaces principal address columns with latitude and longitude.

Usage:
    python3 geocode_businesses.py

The script saves progress to a checkpoint file so it can be safely interrupted
and resumed without re-geocoding already-processed rows.
"""

import csv
import io
import os
import time
import requests
import pandas as pd

INPUT_FILE = "Business_Entities_in_Colorado_20260306 (1).csv"
OUTPUT_FILE = "Business_Entities_in_Colorado_Geocoded.csv"
CHECKPOINT_FILE = "geocode_checkpoint.txt"

BATCH_SIZE = 9999  # Census API max per request
CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"

ADDRESS_COLS = ["principaladdress1", "principaladdress2", "principalcity",
                "principalstate", "principalzipcode", "principalcountry"]


def load_checkpoint():
    """Return the last fully processed row index (0-based, excluding header)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(row_index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(row_index))


def build_census_batch(rows):
    """
    rows: list of (original_index, row_dict)
    Returns a CSV string in the format Census expects:
        Unique ID, Street address, City, State, ZIP
    """
    lines = []
    for idx, row in rows:
        street = (row.get("principaladdress1", "") or "").strip()
        if row.get("principaladdress2"):
            street += " " + row["principaladdress2"].strip()
        city  = (row.get("principalcity", "") or "").strip()
        state = (row.get("principalstate", "") or "").strip()
        zipcode = (row.get("principalzipcode", "") or "").strip()
        # Escape quotes/commas inside fields
        parts = [str(idx), street, city, state, zipcode]
        lines.append(",".join(f'"{p}"' for p in parts))
    return "\n".join(lines)


def geocode_batch(rows):
    """
    Send a batch to the Census geocoder.
    Returns a dict: original_index -> (lat, lon) or (None, None) if unmatched.
    """
    payload_str = build_census_batch(rows)
    payload_bytes = payload_str.encode("utf-8")

    for attempt in range(3):
        try:
            resp = requests.post(
                CENSUS_URL,
                files={"addressFile": ("batch.csv", io.BytesIO(payload_bytes), "text/csv")},
                data={"benchmark": "Public_AR_Current", "vintage": "Current_Current"},
                timeout=120,
            )
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                # Return all unmatched on persistent failure
                return {idx: (None, None) for idx, _ in rows}

    results = {}
    reader = csv.reader(io.StringIO(resp.text))
    for parts in reader:
        if len(parts) < 6:
            continue
        try:
            orig_idx = int(parts[0].strip())
        except ValueError:
            continue
        match = parts[2].strip() if len(parts) > 2 else ""
        coords = parts[5].strip() if len(parts) > 5 else ""
        if match.lower() == "match" and coords:
            try:
                lon_str, lat_str = coords.split(",")
                results[orig_idx] = (float(lat_str.strip()), float(lon_str.strip()))
            except Exception:
                results[orig_idx] = (None, None)
        else:
            results[orig_idx] = (None, None)

    # Ensure every submitted index has an entry
    for idx, _ in rows:
        if idx not in results:
            results[idx] = (None, None)

    return results


def main():
    start_row = load_checkpoint()

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False, low_memory=False)
    total = len(df)
    print(f"Total rows: {total:,}")

    # Add lat/lon columns if they don't exist
    if "latitude" not in df.columns:
        df.insert(df.columns.get_loc("principaladdress1"), "latitude", "")
    if "longitude" not in df.columns:
        df.insert(df.columns.get_loc("principaladdress1"), "longitude", "")

    # Drop address columns (we'll do this after geocoding to avoid losing data mid-run)
    # They stay in the df until we write the final output.

    if start_row > 0:
        print(f"Resuming from row {start_row:,} (already processed {start_row:,} rows).")
        # Load any partial output written so far
        if os.path.exists(OUTPUT_FILE):
            partial = pd.read_csv(OUTPUT_FILE, dtype=str, keep_default_na=False, low_memory=False)
            df.loc[:start_row - 1, "latitude"]  = partial["latitude"].values[:start_row]
            df.loc[:start_row - 1, "longitude"] = partial["longitude"].values[:start_row]

    row_idx = start_row
    while row_idx < total:
        batch_end = min(row_idx + BATCH_SIZE, total)
        batch_rows = [
            (i, df.iloc[i].to_dict())
            for i in range(row_idx, batch_end)
        ]

        print(f"Geocoding rows {row_idx:,}–{batch_end - 1:,} ({batch_end - row_idx} addresses)...", end=" ", flush=True)
        t0 = time.time()
        geo = geocode_batch(batch_rows)
        elapsed = time.time() - t0

        matched = sum(1 for v in geo.values() if v[0] is not None)
        print(f"matched {matched}/{len(batch_rows)} in {elapsed:.1f}s")

        for i, (lat, lon) in geo.items():
            df.at[i, "latitude"]  = str(lat) if lat is not None else ""
            df.at[i, "longitude"] = str(lon) if lon is not None else ""

        row_idx = batch_end
        save_checkpoint(row_idx)

        # Write progress to output file after each batch
        out_df = df.copy()
        out_df = out_df.drop(columns=ADDRESS_COLS, errors="ignore")
        out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. Output written to {OUTPUT_FILE}")
    print(f"Removing checkpoint file.")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()

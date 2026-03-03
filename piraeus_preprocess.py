import pandas as pd
import numpy as np
import pickle
import glob
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(script_dir), "data")
YEARS_TRAIN = [2017, 2018]
YEARS_TEST = [2019]

OUTPUT_DIR = os.path.join(DATA_ROOT, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Trajectory Splitting Parameters
# ==========================================
MAX_TIME_GAP = 1800  # seconds (30 minutes). If gap > 30 mins, start new trajectory.
MIN_SEQ_LEN = 20     # Minimum number of points to constitute a valid trajectory
MIN_SPEED = 0.5      # Knots. Filter out stationary vessels (anchored/moored) to reduce noise.

# ==========================================
# Bounding Box (Greek Seas)
# ==========================================
LAT_MIN, LAT_MAX = 34.0, 42.0
LON_MIN, LON_MAX = 19.0, 30.0

# GLOBAL MAPPING FOR ANONYMIZED MMSI STRINGS
# Keeps track so the same vessel gets the same ID in 2017 and 2019
MMSI_MAP = {}

def encode_vessel_ids(df):
    """
    Converts string hashed MMSIs (e.g., '00168...') to integer IDs.
    Updates the global MMSI_MAP to ensure consistency across years.
    """
    global MMSI_MAP
    
    unique_hashes = df['mmsi'].unique()
    
    new_hashes = [h for h in unique_hashes if h not in MMSI_MAP]
    
    # Assign new Integer IDs starting from the last used ID
    start_id = len(MMSI_MAP) + 1 
    new_mapping = {h: (i + start_id) for i, h in enumerate(new_hashes)}
    
    MMSI_MAP.update(new_mapping)
    
    # Apply mapping to the dataframe
    df['mmsi'] = df['mmsi'].map(MMSI_MAP)
    
    print(f"    Mapped {len(new_hashes)} new vessels. Total unique vessels tracked: {len(MMSI_MAP)}")
    return df

def read_piraeus_csv(file_path):
    """
    Smartly reads a CSV by detecting column names.
    Handles 't' vs 'timestamp', 'vessel_id' vs 'mmsi', etc.
    """
    # check headers
    try:
        header = pd.read_csv(file_path, nrows=0).columns.tolist()
    except Exception as e:
        print(f"    Cannot read header of {file_path}: {e}")
        return None

    # standard_internal_name : [list of possible csv headers]
    column_candidates = {
        'timestamp': ['t', 'timestamp', 'Timestamp', 'time'],
        'mmsi':      ['vessel_id', 'mmsi', 'MMSI', 'id'],
        'lon':       ['lon', 'longitude', 'Lon', 'Longitude'],
        'lat':       ['lat', 'latitude', 'Lat', 'Latitude'],
        'sog':       ['speed', 'sog', 'Speed', 'SOG'],
        'cog':       ['course', 'cog', 'Course', 'COG']
    }

    use_cols = []
    rename_map = {}

    # Match CSV columns to internal standards for unified dataset
    for standard, candidates in column_candidates.items():
        match = next((c for c in candidates if c in header), None)
        if match:
            use_cols.append(match)
            rename_map[match] = standard
        else:
            # If a critical column is missing, skip this file
            raise ValueError(f"Missing required column '{standard}'. Found: {header}")

    df = pd.read_csv(file_path, usecols=use_cols)
    df = df.rename(columns=rename_map)
    return df

def load_and_standardize_year(year):
    # Search pattern
    search_path = os.path.join(DATA_ROOT, f"unipi_ais_dynamic_{year}", f"unipi_ais_dynamic_*{year}.csv")
    files = glob.glob(search_path)
    
    if not files:
        fallback_path = os.path.join(DATA_ROOT, f"unipi_ais_dynamic_{year}", "*.csv")
        files = glob.glob(fallback_path)
    
    if not files:
        print(f"    No files found for year {year} at {search_path}")
        return pd.DataFrame()

    print(f"--> Loading {len(files)} files for year {year}...")
    
    df_list = []
    for f in files:
        try:
            temp_df = read_piraeus_csv(f)
            if temp_df is not None:
                df_list.append(temp_df)
        except Exception as e:
            print(f"    Skipping {os.path.basename(f)}: {e}")

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    # Conversion: ms to s (Check if data is in ms or s first)
    # Piraeus dataset standard is ms
    if df['timestamp'].iloc[0] > 1e11:
         df['timestamp'] = df['timestamp'] / 1000.0

    # Filter
    initial_count = len(df)
    df = df[
        (df['lat'].between(LAT_MIN, LAT_MAX)) & 
        (df['lon'].between(LON_MIN, LON_MAX)) &
        (df['sog'] > MIN_SPEED)
    ]
    df = df.dropna()

    print(f"    Year {year}: Loaded {initial_count} rows. After filtering: {len(df)} rows.")

    if not df.empty:
        df = encode_vessel_ids(df)

    return df

def create_trajectories(df):
    """
    Splits the dataframe into a list of numpy arrays (trajectories)
    based on MMSI and time gaps.
    """
    print("--> Grouping by MMSI and splitting trajectories...")
    
    # Sort is critical for trajectory logic
    df = df.sort_values(by=['mmsi', 'timestamp'])
    
    dataset = []
    
    # Group by vessel
    for mmsi, group in df.groupby('mmsi'):
        # Calculate time difference between consecutive points by resetting index
        group = group.reset_index(drop=True)
        dt = group['timestamp'].diff()    

        # Create traj_id
        # New traj if dt > MAX_TIME_GAP (1800s)
        split_mask = (dt > MAX_TIME_GAP)
        group['traj_id'] = split_mask.cumsum() # cumsum() createus unique ID for each continuous segment

        # Extract segments
        for _, traj in group.groupby('traj_id'):
            if len(traj) >= MIN_SEQ_LEN:
                # Order: [LAT, LON, SOG, COG, TIMESTAMP, MMSI]
                traj_numpy = traj[['lat', 'lon', 'sog', 'cog', 'timestamp', 'mmsi']].to_numpy(dtype=np.float32)
                dataset.append(traj_numpy)
                
    return dataset

def save_dataset(dataset, name):
    """
    Saves the list of numpy arrays to a pickle file.
    """
    path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
    print(f"--> Saving {name} dataset ({len(dataset)} trajectories) to {path}...")
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print("    Done.")

def main():
    print("=== Piraeus AIS Preprocessing for TrAISformer ===")
    
    # --- PROCESS TRAIN SET (2017 + 2018) ---
    train_trajs = []
    for year in YEARS_TRAIN:
        df = load_and_standardize_year(year)
        if not df.empty:
            trajs = create_trajectories(df)
            train_trajs.extend(trajs)
            del df, trajs # Free memory
            
    if train_trajs:
        print(f"\n[Sanity Check] Traj 0 shape: {train_trajs[0].shape}")
        # Verify MMSI is now a number (last column)
        print(f"[Sanity Check] Traj 0 MMSI (encoded): {train_trajs[0][0, 5]}")
        save_dataset(train_trajs, "piraeus_train")
    else:
        print("Error: No training trajectories generated.")

    # --- PROCESS TEST/VAL SET (2019) ---
    test_trajs = []
    for year in YEARS_TEST:
        df = load_and_standardize_year(year)
        if not df.empty:
            trajs = create_trajectories(df)
            test_trajs.extend(trajs)
            del df

    if test_trajs:
        np.random.shuffle(test_trajs)
        split_idx = int(len(test_trajs) * 0.5)
        
        save_dataset(test_trajs[:split_idx], "piraeus_val")
        save_dataset(test_trajs[split_idx:], "piraeus_test")
        
    print("\n=== Preprocessing Complete ===")

if __name__ == "__main__":
    main()
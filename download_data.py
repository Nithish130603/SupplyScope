"""
Download the Favorita Store Sales dataset from Kaggle.
Uses kagglehub for clean, authenticated downloads.
"""
import kagglehub
import shutil
import os

# Download the competition dataset
print("Downloading Favorita Store Sales dataset...")
path = kagglehub.competition_download("store-sales-time-series-forecasting")
print(f"Downloaded to: {path}")

# Copy the key files to our data/ directory
os.makedirs("data", exist_ok=True)

files_needed = [
    "train.csv",         # Daily sales per product family per store (2013-2017)
    "stores.csv",        # Store metadata (city, state, type, cluster)
    "oil.csv",           # Daily oil prices (Ecuador is oil-dependent)
    "holidays_events.csv", # National, regional, and local holidays
    "transactions.csv",  # Daily transaction counts per store
]

for f in files_needed:
    src = os.path.join(path, f)
    dst = os.path.join("data", f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  ✓ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {f} not found")

print("\nDone! Files saved to data/")

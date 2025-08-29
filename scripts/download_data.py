# download_data.py — placeholder for fetching public datasets
# Edit to implement actual downloads (ERA5/GLDAS links, etc.).
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/sample")
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.txt").write_text("Put tiny sample files here for quick tests.", encoding="utf-8")
    print(f"Prepared sample data folder at {dest}")

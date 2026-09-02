KVQ4/NP5/qwen3-32b-q8_deepseek-v3.1-terminus-q4_k_m/results_avg.csv

from pathlib import Path
import pandas as pd

# Root directory to search recursively
root_dir = Path("/path/to/root/folder")

# Output file
output_file = root_dir / "combined.csv"

# Find all CSV files recursively, excluding the output file itself
csv_files = [
    f for f in root_dir.rglob("*.csv")
    if f != output_file
]

print(f"Found {len(csv_files)} CSV files")

dfs = []

for file in csv_files:
    print(f"Reading: {file}")
    df = pd.read_csv(file)

    # Optional: keep track of source file
    df["source_file"] = str(file.relative_to(root_dir))

    dfs.append(df)

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    print(f"\nCombined {len(csv_files)} files")
    print(f"Total rows: {len(combined_df)}")
    print(f"Saved to: {output_file}")
else:
    print("No CSV files found.")

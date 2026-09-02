from pathlib import Path
import pandas as pd

root_dir = Path("/path/to/root_dir")
output_file = root_dir / "combined.csv"

csv_files = list(root_dir.rglob("results_avg.csv"))

dfs = []

for file in csv_files:
    # Example relative path:
    # ollama/KVQ4/NP5/qwen3-32b-q8_deepseek-v3.1-terminus-q4_k_m/results_avg.csv
    parts = file.relative_to(root_dir).parts

    kvq = parts[1]   # KVQ4
    np_value = parts[2]  # NP5

    df = pd.read_csv(file)

    df["KVQ"] = kvq
    df["NP"] = np_value
    df["source_file"] = str(file.relative_to(root_dir))

    dfs.append(df)

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Combined {len(csv_files)} files")
    print(f"Total rows: {len(combined_df)}")
    print(f"Saved to: {output_file}")
else:
    print("No results_avg.csv files found.")

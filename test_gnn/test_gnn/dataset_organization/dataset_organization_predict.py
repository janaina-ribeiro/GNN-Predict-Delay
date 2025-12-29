import csv
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"C:\Users\janaina\OneDrive\Documentos\GNN-Model-Predict")
DELAY_DIR = BASE_DIR / "datasets_atraso_2024"
TRACE_DIR = BASE_DIR / "datasets_traceroute_2024"
OUTPUT_DIR = BASE_DIR / "datasets_generated_prediction"

OUTPUT_DIR.mkdir(exist_ok=True)


def extract_pair_from_filename(filename):
    match = re.search(r"data ([a-z]{2}-[a-z]{2}) ", filename)
    return match.group(1) if match else None


def load_csv_manual(file_path):
    rows = []
    max_cols = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = list(csv.reader([line]))[0]
                max_cols = max(max_cols, len(row))
            except Exception:
                continue

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = list(csv.reader([line]))[0]
                while len(row) < max_cols:
                    row.append("")
                rows.append(row)
            except Exception as e:
                print(f"    Warning: Skipping malformed line {line_num}: {str(e)[:50]}")
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def process_pair(pair_name, delay_file, trace_file):
    print(f"Processing pair: {pair_name}")

    try:
        df_delay = pd.read_csv(delay_file)
        df_delay["Data"] = pd.to_datetime(df_delay["Data"])

        df_trace = None
        loading_methods = [
            lambda: pd.read_csv(trace_file, header=None),
            lambda: pd.read_csv(
                trace_file, header=None, on_bad_lines="skip", engine="python"
            ),
            lambda: pd.read_csv(
                trace_file, header=None, on_bad_lines="warn", engine="python"
            ),
            lambda: pd.read_csv(
                trace_file, header=None, sep=",", quoting=1, engine="python"
            ),
            lambda: load_csv_manual(trace_file),
        ]

        for i, method in enumerate(loading_methods):
            try:
                df_trace = method()
                if df_trace is not None and not df_trace.empty:
                    if i > 0:
                        print(
                            f"  Warning: Used alternative loading method #{i + 1} for {trace_file.name}"
                        )
                    break
            except Exception as method_error:
                if i == len(loading_methods) - 1:
                    raise method_error
                continue

        if df_trace is None or df_trace.empty:
            raise ValueError(
                f"Could not load file {trace_file.name} with any method"
            )

        df_trace["ts"] = pd.to_datetime(df_trace[1])

        link_rows = []

        for idx, row in df_trace.iterrows():
            ts = row["ts"]

            gateway_col = None
            for col_idx in range(len(row)):
                if (
                    pd.notna(row.iloc[col_idx])
                    and str(row.iloc[col_idx]).strip().lower() == "gateway"
                ):
                    gateway_col = col_idx
                    break

            if gateway_col is None:
                gateway_col = 3

            start_col = gateway_col + 1

            ips = []
            hostnames = []

            col_idx = start_col
            while col_idx < len(row):
                ip_val = row.iloc[col_idx] if col_idx < len(row) else None
                hostname_val = (
                    row.iloc[col_idx + 1] if (col_idx + 1) < len(row) else None
                )

                if (
                    pd.isna(ip_val)
                    or str(ip_val).strip() == ""
                    or str(ip_val).strip().lower() == "nan"
                ):
                    break

                ip_clean = str(ip_val).strip().replace("'", "")
                hostname_clean = (
                    str(hostname_val).strip().replace("'", "")
                    if pd.notna(hostname_val)
                    else "No Hostname"
                )

                if ip_clean != "No Ip" and ip_clean != "":
                    ips.append(ip_clean)
                    hostnames.append(hostname_clean)

                col_idx += 2

            if len(ips) < 2:
                continue

            ip_hop_start = ips[0]
            ip_hop_end = ips[-1]

            path_ips = " -> ".join(ips)
            path_hostnames = " -> ".join(hostnames)

            link_rows.append(
                {
                    "Timestamp": ts,
                    "Hop_Start": ip_hop_start,
                    "Hop_End": ip_hop_end,
                    "Hop_Start_Hostname": hostnames[0],
                    "Hop_End_Hostname": hostnames[-1],
                    "Num_Hops": len(ips),
                    "Path_IPs": path_ips,
                    "Path_Hostnames": path_hostnames,
                    "Total_Hops": len(ips),
                    "IP_Hop_Start": ip_hop_start,
                    "IP_Hop_End": ip_hop_end,
                }
            )

        df_links = pd.DataFrame(link_rows)

        df_links_sorted = df_links.sort_values("Timestamp")
        df_delay_sorted = df_delay.sort_values("Data")

        merged = pd.merge_asof(
            df_links_sorted,
            df_delay_sorted[["Data", "Atraso(ms)"]],
            left_on="Timestamp",
            right_on="Data",
            direction="nearest",
            tolerance=pd.Timedelta("80min"),
        )

        # Removed 'high_delay' column as requested

        output_file = OUTPUT_DIR / f"dataset_{pair_name}_links_hops.csv"
        merged.to_csv(output_file, index=False)

        print(f"   File generated: {output_file}")
        print(f"   Total records processed: {len(merged)}")
        print(f"   Traceroutes processed: {len(df_links)}")
        print(f"   Columns in dataset: {list(merged.columns)}")

        return True

    except Exception as e:
        print(f"   Error processing {pair_name}: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        if "Expected" in str(e) and "fields" in str(e):
            print(
                f"   Suggestion: File {trace_file.name} has inconsistent column format"
            )
        return False


print("Starting processing of all pairs...")
print(f"Input folder (delay): {DELAY_DIR}")
print(f"Input folder (traceroute): {TRACE_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print("-" * 60)

delay_files = list(DELAY_DIR.glob("atraso esmond data *.csv"))
total_files = len(delay_files)
processed = 0
errors = 0

for delay_file in delay_files:
    pair_name = extract_pair_from_filename(delay_file.name)

    if not pair_name:
        print(f"Could not extract pair from file: {delay_file.name}")
        errors += 1
        continue

    trace_candidates = sorted(
        TRACE_DIR.glob(f"traceroute esmond data {pair_name} *.csv")
    )

    if not trace_candidates:
        print(
            f"No traceroute file found for {pair_name} in {TRACE_DIR}"
        )
        errors += 1
        continue

    trace_file = trace_candidates[-1]

    print(f"Using traceroute file: {trace_file.name}")

    if process_pair(pair_name, delay_file, trace_file):
        processed += 1
    else:
        errors += 1

print("-" * 60)
print("Processing completed!")
print(f"Total files found: {total_files}")
print(f"Pairs processed successfully: {processed}")
print(f"Errors: {errors}")
print(f"Datasets generated in: {OUTPUT_DIR}")

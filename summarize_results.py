import pandas as pd
import numpy as np

def generate_object_summary(input_csv, output_txt):
    # Load CSV file
    df = pd.read_csv(input_csv)

    # Group by object_name
    grouped = df.groupby('object_name')

    with open(output_txt, 'w') as f:
        for object_name, group in grouped:
            icp_avg = group['icp_score'].mean()
            icp_median = group['icp_score'].median()

            precision_avg = group['precision'].mean() * 100  # Convert to percentage
            precision_median = group['precision'].median() * 100

            time_avg = group['processing_time_ms'].mean()
            time_median = group['processing_time_ms'].median()

            f.write(f"--- {object_name} ---\n")
            f.write(f"ICP Score      → Average: {icp_avg:.6f}, Median: {icp_median:.6f}\n")
            f.write(f"Precision (%)  → Average: {precision_avg:.2f}, Median: {precision_median:.2f}\n")
            f.write(f"Time (ms)      → Average: {time_avg:.2f}, Median: {time_median:.2f}\n\n")

# Usage
generate_object_summary('recognition_results.csv', 'result_summary.txt')

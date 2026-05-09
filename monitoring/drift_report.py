from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Dataset path
DATASET_PATH = "data/telco_churn.csv"

# Output report
REPORT_DIR = Path("monitoring/reports")
REPORT_PATH = REPORT_DIR / "drift_report.html"


def main():
    # Create reports directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Simulate drift:
    # split dataset into reference/current parts
    split_index = int(len(df) * 0.5)

    reference_df = df.iloc[:split_index].copy()
    current_df = df.iloc[split_index:].copy()

    # Example artificial drift
    if "MonthlyCharges" in current_df.columns:
        current_df["MonthlyCharges"] = (
            current_df["MonthlyCharges"] * 1.15
        )

    # Create report
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    # Run report
    evaluation = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    # Save report
    evaluation.save_html(str(REPORT_PATH))

    print(f"Drift report generated: {REPORT_PATH}")


if __name__ == "__main__":
    main()
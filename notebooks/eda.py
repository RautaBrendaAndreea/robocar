"""EDA script - Run to validate data quality before training."""
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import seaborn as sns
from src.collector.data_collector import DataCollector


def run_eda(data_dir="data"):
    df = DataCollector.load_all(data_dir)
    ray_cols = [c for c in df.columns if c.startswith("ray_")]

    print(f"Total samples: {len(df)}")
    print(f"Features: {len(ray_cols)} rays + 2 actions")
    print(f"\n--- Stats ---\n{df.describe()}")
    print(f"\n--- Missing values ---\n{df.isnull().sum().sum()} total")
    print(f"\n--- Action distribution ---")
    print(f"  Throttle: mean={df['throttle'].mean():.3f}, std={df['throttle'].std():.3f}")
    print(f"  Steering: mean={df['steering'].mean():.3f}, std={df['steering'].std():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    df[["throttle", "steering"]].hist(ax=axes[0], bins=50)
    axes[0][0].set_title("Throttle Distribution")
    axes[0][1].set_title("Steering Distribution")

    axes[1][0].plot(df["throttle"].values[:500], label="throttle", alpha=0.7)
    axes[1][0].plot(df["steering"].values[:500], label="steering", alpha=0.7)
    axes[1][0].set_title("Actions over time (first 500)")
    axes[1][0].legend()

    sns.heatmap(df[ray_cols].corr(), ax=axes[1][1], cmap="coolwarm", xticklabels=False, yticklabels=False)
    axes[1][1].set_title("Ray correlations")

    plt.tight_layout()
    plt.savefig("notebooks/eda_report.png", dpi=150)
    plt.show()
    print("\nEDA report saved to notebooks/eda_report.png")


if __name__ == "__main__":
    run_eda()

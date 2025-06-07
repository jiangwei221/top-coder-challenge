import pandas as pd
import matplotlib.pyplot as plt
import json

with open('public_cases.json') as fh:
    rows = json.load(fh)

# Flatten JSON → DataFrame
df = pd.json_normalize(rows)

# Keep just day-1 records and rename columns for convenience
subset = (
    df[df["input.trip_duration_days"] == 4]
    .rename(
        columns={
            "input.miles_traveled": "miles_traveled",
            "input.total_receipts_amount": "total_receipts_amount",
        }
    )[["miles_traveled", "total_receipts_amount", "expected_output"]]
)

outliers = subset[(subset["total_receipts_amount"] > 1500) & (subset["expected_output"] < 600)]
print("Outlier rows:")
print(outliers)

# --- Examine records with 1500 < total_receipts_amount < 2000 ---------------
mid_receipts = subset[
    (subset["total_receipts_amount"] > 1500) & (subset["total_receipts_amount"] < 2000)
]
print("\nRows with 1500 < total_receipts_amount < 2000:")
print(mid_receipts)

# Visualise how expected_output changes for this band of receipts
plt.figure(figsize=(6, 4))
plt.scatter(
    mid_receipts["miles_traveled"],
    mid_receipts["expected_output"],
    color="orange",
    s=80,
)
plt.xlabel("Miles traveled")
plt.ylabel("Expected output")
plt.title("Expected output vs miles (1500 < receipts < 2000, days = 1)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2×2 scatter-matrix
pd.plotting.scatter_matrix(
    subset,
    figsize=(7, 7),
    diagonal="hist",
    alpha=0.7,
)
plt.suptitle("Pairwise relationships (trip_duration_days = 1)", y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sc = plt.scatter(
    subset["miles_traveled"],
    subset["expected_output"],
    c=subset["total_receipts_amount"],
    cmap="viridis",
    s=60,
)
plt.colorbar(sc, label="total_receipts_amount")
plt.xlabel("Miles traveled")
plt.ylabel("Expected output")
plt.title("Expected output vs miles (color = receipts, days = 1)")
plt.grid(True)
plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    subset["miles_traveled"],
    subset["total_receipts_amount"],
    subset["expected_output"],
    depthshade=True,
)
ax.set_xlabel("Miles traveled")
ax.set_ylabel("Total receipts ($)")
ax.set_zlabel("Expected output")
ax.set_title("3-D relation at trip_duration_days = 1")
plt.tight_layout()
plt.show()


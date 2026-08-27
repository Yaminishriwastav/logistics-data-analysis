import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------------------------------
# 1. Create 500 simulated shipments
# --------------------------------------------------

n_samples = 500

transport_modes = np.random.choice(
    ["Road", "Air", "Sea", "Rail"],
    size=n_samples,
    p=[0.45, 0.15, 0.25, 0.15]
)

regions = np.random.choice(
    ["North", "South", "East", "West", "Central"],
    size=n_samples
)

distance = np.random.gamma(
    shape=5,
    scale=120,
    size=n_samples
).round(1)

weight = np.random.uniform(
    10,
    2000,
    size=n_samples
).round(1)

# Average speed assumptions
speed = {
    "Road": 55,
    "Air": 650,
    "Sea": 35,
    "Rail": 70
}

# Cost rate assumptions
rate = {
    "Road": 1.10,
    "Air": 4.00,
    "Sea": 0.55,
    "Rail": 0.85
}

# --------------------------------------------------
# 2. Generate delivery time
# --------------------------------------------------

delivery_time = np.array([
    distance[i]
    / speed[transport_modes[i]]
    * 1.4
    + np.random.normal(0, 1.2)

    for i in range(n_samples)
])

delivery_time = np.maximum(
    delivery_time,
    1
).round(2)

# --------------------------------------------------
# 3. Generate transport cost
# --------------------------------------------------

transport_cost = np.array([
    distance[i]
    * rate[transport_modes[i]]
    + weight[i] * 0.12
    + np.random.normal(0, 50)

    for i in range(n_samples)
])

transport_cost = np.maximum(
    transport_cost,
    50
).round(2)

# --------------------------------------------------
# 4. Generate delay and carrier rating
# --------------------------------------------------

delay = np.random.exponential(
    scale=45,
    size=n_samples
).round(1)

rating = (
    5
    - (delay / 220)
    + np.random.normal(
        0,
        0.18,
        n_samples
    )
)

rating = np.clip(
    rating,
    1,
    5
).round(2)

# --------------------------------------------------
# 5. Create final dataframe
# --------------------------------------------------

df = pd.DataFrame({
    "Shipment_ID": [
        f"SHP{10000+i}"
        for i in range(n_samples)
    ],

    "Transport_Mode": transport_modes,

    "Region": regions,

    "Distance_KM": distance,

    "Weight_KG": weight,

    "Delivery_Time_Hours":
        delivery_time,

    "Transport_Cost_USD":
        transport_cost,

    "Delay_Minutes": delay,

    "Carrier_Rating": rating
})

df.to_csv(
    "week3_logistics_500_shipments.csv",
    index=False
)

# --------------------------------------------------
# 6. Summary statistics
# --------------------------------------------------

numeric_columns = [
    "Distance_KM",
    "Weight_KG",
    "Delivery_Time_Hours",
    "Transport_Cost_USD",
    "Delay_Minutes",
    "Carrier_Rating"
]

print("WEEK 3 - SUMMARY STATISTICS")
print("-" * 40)

print(
    df[numeric_columns]
    .describe()
    .round(2)
)

# --------------------------------------------------
# 7. Cost by transport mode
# --------------------------------------------------

print("\nAverage cost by transport mode:")

print(
    df.groupby("Transport_Mode")
    ["Transport_Cost_USD"]
    .mean()
    .round(2)
)

# --------------------------------------------------
# 8. Delay by transport mode
# --------------------------------------------------

print("\nAverage delay by transport mode:")

print(
    df.groupby("Transport_Mode")
    ["Delay_Minutes"]
    .mean()
    .round(2)
)

# --------------------------------------------------
# 9. Cost vs distance
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for mode in [
    "Road",
    "Air",
    "Sea",
    "Rail"
]:

    subset = df[
        df["Transport_Mode"] == mode
    ]

    plt.scatter(
        subset["Distance_KM"],
        subset["Transport_Cost_USD"],
        label=mode,
        alpha=0.65
    )

plt.xlabel("Distance (KM)")
plt.ylabel("Transport Cost (USD)")
plt.title("Transport Cost vs Distance")
plt.legend()
plt.tight_layout()

plt.savefig(
    "01_cost_vs_distance.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# 10. Delay distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

df.boxplot(
    column="Delay_Minutes",
    by="Transport_Mode"
)

plt.suptitle("")
plt.title(
    "Shipment Delay Distribution "
    "by Transport Mode"
)

plt.xlabel("Transport Mode")
plt.ylabel("Delay (Minutes)")

plt.tight_layout()

plt.savefig(
    "02_delay_distribution.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# 11. Correlation matrix
# --------------------------------------------------

plt.figure(figsize=(8, 6))

correlation = df[
    numeric_columns
].corr()

plt.imshow(correlation)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.colorbar(
    label="Correlation"
)

plt.title(
    "Correlation Matrix of Logistics Metrics"
)

plt.tight_layout()

plt.savefig(
    "03_correlation_matrix.png",
    dpi=150
)

plt.show()

print("\nCorrelation matrix:")
print(correlation.round(2))

# --------------------------------------------------
# 12. Regional performance
# --------------------------------------------------

regional = df.groupby("Region").agg(
    Average_Cost=(
        "Transport_Cost_USD",
        "mean"
    ),

    Average_Delivery_Time=(
        "Delivery_Time_Hours",
        "mean"
    )
)

regional.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Regional Cost and Delivery Time"
)

plt.xlabel("Region")
plt.ylabel("Average Value")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "04_regional_performance.png",
    dpi=150
)

plt.show()

print("\nRegional performance:")
print(regional.round(2))

# --------------------------------------------------
# 13. Practical findings
# --------------------------------------------------

highest_cost_mode = (
    df.groupby("Transport_Mode")
    ["Transport_Cost_USD"]
    .mean()
    .idxmax()
)

lowest_delay_mode = (
    df.groupby("Transport_Mode")
    ["Delay_Minutes"]
    .median()
    .idxmin()
)

highest_cost_region = (
    df.groupby("Region")
    ["Transport_Cost_USD"]
    .mean()
    .idxmax()
)

delay_rating_correlation = df[
    ["Delay_Minutes", "Carrier_Rating"]
].corr().iloc[0, 1]

print("\nPRACTICAL FINDINGS")
print("-" * 30)

print(
    "Highest average-cost mode:",
    highest_cost_mode
)

print(
    "Lowest median-delay mode:",
    lowest_delay_mode
)

print(
    "Highest average-cost region:",
    highest_cost_region
)

print(
    "Delay vs carrier-rating correlation:",
    round(delay_rating_correlation, 2)
)

print("\nRecommendations:")
print(
    "1. Compare transport modes before selecting routes."
)

print(
    "2. Monitor delays and investigate severe cases."
)

print(
    "3. Track carrier ratings together with delivery delays."
)

print(
    "4. Review regional cost and delivery-time patterns."
)

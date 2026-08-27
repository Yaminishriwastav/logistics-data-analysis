import numpy as np
import pandas as pd

np.random.seed(42)

# Create a sample logistics dataset
n = 100

df = pd.DataFrame({
    "Shipment_ID": [f"SHP{10000+i}" for i in range(n)],
    "Transport_Mode": np.random.choice(
        ["Road", "Air", "Sea", "Rail"],
        n,
        p=[0.45, 0.15, 0.25, 0.15]
    ),
    "Distance_KM": np.random.gamma(
        5, 120, n
    ).round(1),
    "Delivery_Time_days": np.random.uniform(
        1, 10, n
    ).round(2),
    "Shipping_Cost": np.random.uniform(
        100, 2500, n
    ).round(2),
    "Delivery_Status": np.random.choice(
        ["On Time", "Delayed"],
        n,
        p=[0.82, 0.18]
    )
})

# Calculate KPIs
average_delivery_time = df["Delivery_Time_days"].mean()
average_shipping_cost = df["Shipping_Cost"].mean()
average_distance = df["Distance_KM"].mean()

on_time_rate = (
    df["Delivery_Status"].eq("On Time").mean()
    * 100
)

print("WEEK 1 - LOGISTICS ANALYSIS")
print("-" * 35)

print("Number of shipments:", len(df))
print(
    f"Average distance: {average_distance:.2f} km"
)
print(
    f"Average delivery time: "
    f"{average_delivery_time:.2f} days"
)
print(
    f"Average shipping cost: "
    f"{average_shipping_cost:.2f}"
)
print(
    f"On-time delivery rate: "
    f"{on_time_rate:.2f}%"
)

print("\nShipments by transport mode:")
print(df["Transport_Mode"].value_counts())

print("\nSuggested analysis areas:")
print("1. Compare transport modes.")
print("2. Study delivery delays.")
print("3. Compare distance and shipping cost.")
print("4. Monitor on-time delivery performance.")

# Save sample data
df.to_csv(
    "week1_logistics_sample.csv",
    index=False
)

print(
    "\nSample dataset saved as "
    "week1_logistics_sample.csv"
)

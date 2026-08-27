import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

# --------------------------------------------------
# 1. Create a sample raw logistics dataset
# --------------------------------------------------

n = 120

df = pd.DataFrame({
    "Shipment_ID": [
        f"SHP{10000+i}" for i in range(n)
    ],

    "Order_Date": pd.date_range(
        "2026-01-01",
        periods=n,
        freq="D"
    ),

    "Delivery_Date": pd.date_range(
        "2026-01-03",
        periods=n,
        freq="D"
    ),

    "Distance_km": np.random.uniform(
        20, 1500, n
    ).round(1),

    "Quantity": np.random.randint(
        1, 50, n
    ),

    "Shipping_Cost": np.random.uniform(
        100, 3000, n
    ).round(2),

    "Delivery_Time_days": np.random.uniform(
        1, 12, n
    ).round(2),

    "Weight_kg": np.random.uniform(
        5, 2000, n
    ).round(1),

    "Delivery_Status": np.random.choice(
        ["On Time", "Delayed"],
        n,
        p=[0.80, 0.20]
    )
})

# Add a few missing values
df.loc[5, "Shipping_Cost"] = np.nan
df.loc[12, "Weight_kg"] = np.nan
df.loc[20, "Delivery_Status"] = np.nan

# Add duplicate rows
df = pd.concat(
    [df, df.iloc[[10, 11]]],
    ignore_index=True
)

# Save raw data
df.to_csv(
    "raw_logistics_data.csv",
    index=False
)

print("Original dataset shape:", df.shape)
print(
    "Duplicate rows:",
    df.duplicated().sum()
)

# --------------------------------------------------
# 2. Remove duplicates
# --------------------------------------------------

df = df.drop_duplicates().copy()

# --------------------------------------------------
# 3. Convert date columns
# --------------------------------------------------

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df["Delivery_Date"] = pd.to_datetime(
    df["Delivery_Date"],
    errors="coerce"
)

# --------------------------------------------------
# 4. Calculate delivery duration
# --------------------------------------------------

df["Calculated_Delivery_Days"] = (
    df["Delivery_Date"]
    - df["Order_Date"]
).dt.days

# --------------------------------------------------
# 5. Handle missing numerical values
# --------------------------------------------------

numeric_columns = [
    "Distance_km",
    "Quantity",
    "Shipping_Cost",
    "Delivery_Time_days",
    "Weight_kg"
]

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# --------------------------------------------------
# 6. Handle missing categorical values
# --------------------------------------------------

df["Delivery_Status"] = (
    df["Delivery_Status"]
    .fillna(df["Delivery_Status"].mode()[0])
)

# --------------------------------------------------
# 7. Detect outliers using IQR
# --------------------------------------------------

q1 = df["Shipping_Cost"].quantile(0.25)
q3 = df["Shipping_Cost"].quantile(0.75)

iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

df["Shipping_Cost_Outlier"] = (
    (df["Shipping_Cost"] < lower_limit)
    |
    (df["Shipping_Cost"] > upper_limit)
)

print(
    "Possible shipping-cost outliers:",
    int(df["Shipping_Cost_Outlier"].sum())
)

# --------------------------------------------------
# 8. Min-Max normalization
# --------------------------------------------------

scale_columns = [
    "Distance_km",
    "Quantity",
    "Shipping_Cost",
    "Delivery_Time_days",
    "Weight_kg"
]

scaler = MinMaxScaler()

df[scale_columns] = scaler.fit_transform(
    df[scale_columns]
)

# --------------------------------------------------
# 9. Save cleaned data
# --------------------------------------------------

df.to_csv(
    "cleaned_logistics_data.csv",
    index=False
)

print("\nWEEK 2 - PREPROCESSING COMPLETED")
print("-" * 35)

print("Rows after cleaning:", len(df))

print(
    "Remaining missing values:",
    int(df.isna().sum().sum())
)

print(
    "Cleaned file saved as "
    "cleaned_logistics_data.csv"
)

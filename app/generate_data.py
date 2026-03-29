import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

products = [
    (1, "iPhone", "Electronics", 999),
    (2, "MacBook", "Electronics", 1500),
    (3, "T-Shirt", "Clothing", 25),
    (4, "Jeans", "Clothing", 60),
    (5, "Milk", "Groceries", 3),
    (6, "Bread", "Groceries", 2),
    (7, "Sneakers", "Footwear", 80),
    (8, "Watch", "Accessories", 120),
]

rows = []
start_date = datetime(2023, 1, 1)

for product_id, name, category, price in products:
    base_demand = np.random.randint(50, 200)

    for day in range(365):  # FULL YEAR = ~3000 rows total
        date = start_date + timedelta(days=day)

        # 📈 Trend (slow growth)
        trend = day * np.random.uniform(0.05, 0.3)

        # 📅 Weekly seasonality
        seasonality = 20 * np.sin(2 * np.pi * day / 7)

        # 🎉 Weekend boost
        weekend = 40 if date.weekday() >= 5 else 0

        # 🎁 Festival spike (random days)
        festival = 100 if np.random.rand() < 0.03 else 0

        # 🎲 Noise
        noise = np.random.normal(0, 10)

        demand = max(10, int(base_demand + trend + seasonality + weekend + festival + noise))

        # 💰 Sales
        sales = demand * price * np.random.uniform(0.9, 1.1)

        # 📊 Profit
        cost_ratio = np.random.uniform(0.6, 0.8)
        profit = sales * (1 - cost_ratio)

        rows.append({
            "product_id": product_id,
            "product_name": name,
            "category": category,
            "date": date.strftime("%Y-%m-%d"),
            "price": price,
            "demand": demand,
            "sales": int(sales),
            "profit": int(profit)
        })

df = pd.DataFrame(rows)

df.to_csv("shop_data.csv", index=False)

print("✅ Dataset generated: shop_data.csv")
print("Rows:", len(df))
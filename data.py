import pandas as pd
import numpy as np

np.random.seed(42)

n = 200  # increase dataset size

area = np.random.randint(500, 4000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 5, n)

# realistic non-linear pricing model
price = (
    (area ** 1.05) * 900 +
    bedrooms * 300000 +
    bathrooms * 200000 +
    np.random.normal(0, 250000, n)
)

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "price": price
})

df.to_csv("data.csv", index=False)
print("Dataset created successfully!")
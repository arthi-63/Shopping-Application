import pandas as pd

# Load products
products = pd.read_csv("data/products.csv")

# Load categories
categories = pd.read_csv("data/amazon_categories.csv")

# Rename column to match products.csv
categories.rename(columns={"id": "category_id"}, inplace=True)

# Merge category names into products
products = products.merge(
    categories,
    on="category_id",
    how="left"
)

# Save updated dataset
products.to_csv("data/products.csv", index=False)

print("Categories merged successfully!")
print(products.head())
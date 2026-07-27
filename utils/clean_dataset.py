import pandas as pd

# Read the original Amazon dataset
df = pd.read_csv("data/amazon_products.csv")

print(f"Original dataset shape: {df.shape}")

# Keep only the columns we need
columns_to_keep = [
    "asin",
    "title",
    "stars",
    "reviews",
    "price",
    "listPrice",
    "category_id",
    "isBestSeller",
    "boughtInLastMonth",
    "imgUrl",
    "productURL"
]

df = df[columns_to_keep]

# Remove rows with missing title or price
df = df.dropna(subset=["title", "price"])

# Remove duplicate products
df = df.drop_duplicates(subset=["asin"])

# Keep only the first 5000 products
df = df.head(5000)

# Save as products.csv
df.to_csv("data/products.csv", index=False)

print(f"Clean dataset shape: {df.shape}")
print("products.csv created successfully!")
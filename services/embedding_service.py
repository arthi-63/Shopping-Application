import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded successfully!")

    def generate_embeddings(self):

        # Read products
        df = pd.read_csv("data/products.csv")

        # Convert each product into meaningful text
        product_texts = []

        for _, row in df.iterrows():

            text = f"""
Product: {row['title']}
Category: {row['category_name']}
Price: {row['price']}
List Price: {row['listPrice']}
Rating: {row['stars']}
Reviews: {row['reviews']}
Best Seller: {row['isBestSeller']}
Bought Last Month: {row['boughtInLastMonth']}
"""

            product_texts.append(text)

        print(f"Generating embeddings for {len(product_texts)} products...")

        embeddings = self.model.encode(
            product_texts,
            show_progress_bar=True
        )

        # Save embeddings
        with open("data/embedding.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        print("Embeddings saved successfully!")


if __name__ == "__main__":

    embedding_service = EmbeddingService()
    embedding_service.generate_embeddings()
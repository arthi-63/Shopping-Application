import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from services.gemini_service import GeminiService


class RecommendationService:

    def __init__(self):

        print("Loading model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading product dataset...")
        self.products = pd.read_csv("data/products.csv")

        print("Loading vector index...")
        self.index = faiss.read_index("data/vector.index")

        print("Loading Gemini service...")
        self.gemini = GeminiService()

        print("Recommendation service ready!")

    def search_products(self, query, top_k=5):

        # Convert user query into embedding
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # Search similar products
        distances, indices = self.index.search(query_embedding, top_k)

        recommendations = []

        for idx in indices[0]:

            product = self.products.iloc[idx]

            recommendations.append({
                "title": product["title"],
                "price": product["price"],
                "rating": product["stars"],
                "reviews": product["reviews"]
            })

        # Generate AI recommendation using Gemini
        ai_response = self.gemini.generate_recommendation(
            query,
            recommendations
        )

        return recommendations, ai_response


if __name__ == "__main__":

    recommender = RecommendationService()

    query = input("Enter your shopping query: ")

    results, ai_response = recommender.search_products(query)

    print("\nRecommended Products\n")

    for i, product in enumerate(results, start=1):

        print(f"{i}. {product['title']}")
        print(f"   Price: {product['price']}")
        print(f"   Rating: {product['rating']}")
        print(f"   Reviews: {product['reviews']}")
        print()

    print("=" * 50)
    print("AI SHOPPING ASSISTANT")
    print("=" * 50)
    print(ai_response)
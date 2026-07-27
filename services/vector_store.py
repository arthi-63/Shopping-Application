import faiss
import pickle
import numpy as np


class VectorStore:

    def __init__(self):

        print("Loading embeddings...")

        with open("data/embedding.pkl", "rb") as f:
            self.embeddings = pickle.load(f)

        self.embeddings = np.array(self.embeddings).astype("float32")

        print("Embeddings loaded successfully!")

    def build_index(self):

        dimension = self.embeddings.shape[1]

        print(f"Embedding Dimension: {dimension}")

        index = faiss.IndexFlatL2(dimension)

        index.add(self.embeddings)

        faiss.write_index(index, "data/vector.index")

        print("Vector index saved successfully!")


if __name__ == "__main__":

    vector_store = VectorStore()
    vector_store.build_index()
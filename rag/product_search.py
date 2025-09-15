import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from shared.config import settings

class ProductSearch:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)


    def search(self, query: str, limit: int = 5) -> list[models.ScoredPoint]:
        results = self.qdrant_client.query_points(
            collection_name=settings.COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=settings.DENSE_EMBEDDING_MODEL,
                    ),
                    using="jina-small",
                    limit=(10 * limit),
                ),
            ],
            query=models.Document(
                text=query,
                model=settings.SPARSE_EMBEDDING_MODEL,
            ),
            using="bm25",
            limit=limit,
            with_payload=True,
        )

        return [p.payload for p in results.points]

    def multi_query_hybrid_search(self, queries: list[str], limit: int = 5) -> list[dict]:
        """
        Perform hybrid vector search for multiple queries. Returns a flattened list of results for all queries.
        """
        all_results = []
        for query in queries:
            results = self.qdrant_client.query_points(
                collection_name=settings.COLLECTION,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model=settings.DENSE_EMBEDDING_MODEL,
                        ),
                        using="jina-small",
                        limit=(10 * limit),
                    ),
                ],
                query=models.Document(
                    text=query,
                    model=settings.SPARSE_EMBEDDING_MODEL,
                ),
                using="bm25",
                limit=limit,
                with_payload=True,
            )
            all_results.extend([p.payload for p in results.points])
        return all_results


if __name__ == "__main__":
    # For testing purposes
    ps = ProductSearch()
    query = "I am a women and need business casual"
    print(f"Query: {query}")
    results = ps.search(query, limit=5)
    for idx, res in enumerate(results):
        print(f"Result {idx + 1}: {res}")
        print()
from tqdm.auto import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from shared.config import settings
from pathlib import Path
import csv

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "myntra_products_catalog.csv"

def load_products(data_file):
    products = []
    with open(data_file, 'r') as f_in:
        reader = csv.DictReader(f_in)
        for row in tqdm(reader):
            product = {
                'product_name': row['ProductName'],
                'gender': row['Gender'],
                'brand': row['ProductBrand'],
                'description': row['Description'],
                'price': row['Price (INR)'],
                'color': row['PrimaryColor'],
                'id': row['ProductID']
            }
            products.append(product)
    print(f"Number of products: {len(products)}")
    return products

def create_collection_if_needed(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "jina-small": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE,
                )},
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        )

def build_points(products, dense_model_handle, sparse_model_handle):
    points = []
    for idx, prod in enumerate(products):
        prod_text = prod['product_name'] + ' ' + prod['description']
        dense_vect = models.Document(text=prod_text, model=dense_model_handle)
        sparse_vect = models.Document(text=prod_text, model=sparse_model_handle)
        v = {"jina-small": dense_vect, "bm25": sparse_vect}
        point = PointStruct(id=idx, vector=v, payload=prod)
        points.append(point)
    print(f"points - {len(points)}")
    return points

def upsert_points(client, collection_name, points):
    client.upsert(
        collection_name=collection_name,
        points=points
    )

def batch_upsert_points(client, collection_name, points, batch_size=2000):
    try:
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            upsert_points(client, collection_name, batch)
            print(f"Upserted points {i} to {i+len(batch)}")
    except Exception as e:
        print(f"Error during upsert: {e}")
    finally:
        print("Upsert process completed.")

def main():
    products = load_products(DATA_FILE)
    client = QdrantClient(url=settings.QDRANT_URL)
    sparse_model_handle = settings.SPARSE_EMBEDDING_MODEL
    dense_model_handle = settings.DENSE_EMBEDDING_MODEL
    create_collection_if_needed(client, settings.COLLECTION)
    points = build_points(products, dense_model_handle, sparse_model_handle)
    print("Starting to insert product data")
    batch_upsert_points(client, settings.COLLECTION, points)

if __name__ == "__main__":
    main()
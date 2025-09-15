from prefect import task, flow
from tqdm.auto import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from shared.config import settings
from pathlib import Path
import csv

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "products_catalog.csv"

@task
def load_products_task(data_file):
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

@task
def create_collection_task(collection_name):
    client = QdrantClient(url=settings.QDRANT_URL)
    if not client.collection_exists(collection_name):
        create_collection(collection_name, client)
    else:
        print(f"Collection {collection_name} already exists. Dropping and recreating.")
        client.delete_collection(collection_name)
        create_collection(collection_name, client)

def create_collection(collection_name, client):
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

@task
def build_points_task(products, dense_model_handle, sparse_model_handle):
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

@task
def batch_upsert_points_task(collection_name, points, batch_size=2000):
    client = QdrantClient(url=settings.QDRANT_URL)
    try:
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"Upserted points {i} to {i+len(batch)}")
    except Exception as e:
        print(f"Error during upsert: {e}")
    finally:
        print("Upsert process completed.")

@flow
def ingest_data_flow():
    products = load_products_task(DATA_FILE)
    create_collection_task(settings.COLLECTION)
    points = build_points_task(products, settings.DENSE_EMBEDDING_MODEL, settings.SPARSE_EMBEDDING_MODEL)
    batch_upsert_points_task(settings.COLLECTION, points)

if __name__ == "__main__":
    ingest_data_flow()
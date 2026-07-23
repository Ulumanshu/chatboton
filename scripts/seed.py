"""Seeds Neo4j (purchase graph) and Chroma (review vectors) with demo data.

Postgres is seeded automatically by docker/postgres-init/ on first boot.
Idempotent: Neo4j uses MERGE, Chroma uses upsert with fixed ids.

Usage: python scripts/seed.py   (docker compose and Ollama must be running)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from opensearchpy import OpenSearch
import requests

from app.config import load_env_file

PURCHASES = [
    ("Ada",   "Vilnius", "Volta Powerbank 20k",   2),
    ("Ada",   "Vilnius", "Nimbus Wireless Mouse", 1),
    ("Bruno", "Kaunas",  "Krakla Mech Keyboard",  1),
    ("Bruno", "Kaunas",  "Aurix 4K Webcam",       1),
    ("Celia", "Riga",    "Sonar BT Speaker",      3),
    ("Celia", "Riga",    "Volta Powerbank 20k",   1),
    ("Dovydas", "Vilnius", "Piksel USB-C Hub",    2),
]

PRODUCT_CATEGORIES = {
    "Volta Powerbank 20k": "power",
    "Nimbus Wireless Mouse": "input",
    "Krakla Mech Keyboard": "input",
    "Aurix 4K Webcam": "video",
    "Sonar BT Speaker": "audio",
    "Piksel USB-C Hub": "power",
}

REVIEWS = [
    ("rev-1", "Volta Powerbank 20k", 5, "Charged my phone four times on one charge, great for hiking trips."),
    ("rev-2", "Volta Powerbank 20k", 3, "Works fine but heavy in the pocket and slow to recharge itself."),
    ("rev-3", "Nimbus Wireless Mouse", 4, "Smooth scrolling and long battery life, though the click is a bit loud."),
    ("rev-4", "Krakla Mech Keyboard", 5, "Fantastic tactile switches, my typing speed went up noticeably."),
    ("rev-5", "Aurix 4K Webcam", 2, "Picture is sharp but the autofocus keeps hunting during video calls."),
    ("rev-6", "Sonar BT Speaker", 4, "Rich bass for its size, perfect for the kitchen and balcony."),
    ("rev-7", "Piksel USB-C Hub", 5, "Finally one hub for HDMI, ethernet and SD cards that does not overheat."),
]


def seed_neo4j():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:57688")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "chatboton_password"))
    with GraphDatabase.driver(uri, auth=auth) as driver:
        for customer, city, product, qty in PURCHASES:
            driver.execute_query(
                "MERGE (c:Customer {name: $customer}) SET c.city = $city "
                "MERGE (p:Product {name: $product}) SET p.category = $category "
                "MERGE (c)-[b:BOUGHT]->(p) SET b.qty = $qty",
                customer=customer, city=city, product=product,
                category=PRODUCT_CATEGORIES[product], qty=qty,
            )
    print(f"Neo4j: seeded {len(PURCHASES)} purchases.")


def seed_chroma():
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "58001")),
    )
    embedder = OllamaEmbeddingFunction(
        url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model_name=os.getenv("CHROMA_EMBED_MODEL", "nomic-embed-text"),
    )
    collection = client.get_or_create_collection(
        name=os.getenv("CHROMA_COLLECTION", "product_reviews"),
        embedding_function=embedder,
    )
    collection.upsert(
        ids=[review_id for review_id, _, _, _ in REVIEWS],
        documents=[text for _, _, _, text in REVIEWS],
        metadatas=[{"product": product, "rating": rating} for _, product, rating, _ in REVIEWS],
    )
    print(f"Chroma: seeded {len(REVIEWS)} reviews.")


def seed_qdrant():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "58333")),
    )
    collection_name = os.getenv("QDRANT_COLLECTION", "products")
    
    # Check if collection exists
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE),
        )

    points = []
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    for i, (name, category) in enumerate(PRODUCT_CATEGORIES.items()):
        try:
            resp = requests.post(
                f"{ollama_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": name}
            )
            vector = resp.json()["embedding"]
            points.append(qmodels.PointStruct(
                id=i,
                vector=vector,
                payload={"name": name, "category": category}
            ))
        except Exception as e:
            print(f"Qdrant: failed to get embedding for {name}: {e}")

    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Qdrant: seeded {len(points)} products.")


def seed_opensearch():
    client = OpenSearch(
        hosts=[{'host': os.getenv("OPENSEARCH_HOST", "localhost"), 
                'port': int(os.getenv("OPENSEARCH_PORT", "59200"))}],
        use_ssl=False,
        verify_certs=False,
    )
    index_name = os.getenv("OPENSEARCH_INDEX", "products")
    
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)

    for i, (name, category) in enumerate(PRODUCT_CATEGORIES.items()):
        doc = {"name": name, "category": category}
        client.index(index=index_name, id=i, body=doc, refresh=True)
    
    print(f"OpenSearch: seeded {len(PRODUCT_CATEGORIES)} products.")


if __name__ == "__main__":
    load_env_file()
    seed_neo4j()
    seed_chroma()
    seed_qdrant()
    seed_opensearch()

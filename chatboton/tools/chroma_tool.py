"""Vector tool: semantic search against the customer reviews collection."""

from langchain_core.tools import tool
from chatboton.connectors.chroma import ChromaConnector

@tool
def search_customer_reviews(query: str, n_results: int = 3) -> str:
    """Semantically searches customer reviews of gadget-store products.

    Reviews live in a Chroma vector store; each hit carries metadata
    {product, rating}. Product names match the Postgres products
    table.

    Args:
        query: Natural-language search text.
        n_results: How many closest reviews to return.

    Returns:
        JSON list of {document, metadata, distance} hits, or an error
        description.
    """
    return ChromaConnector(collection="product_reviews").query(query, n_results)

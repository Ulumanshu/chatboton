"""Search tool: keyword search in OpenSearch docs."""

from langchain_core.tools import tool
from chatboton.connectors.opensearch import OpenSearchConnector

@tool
def full_text_search_docs(query: str) -> str:
    """Performs full-text search on product names and categories in OpenSearch.
    
    Args:
        query: The search term.
    """
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["name", "category"]
            }
        }
    }
    return OpenSearchConnector(collection="products").query(body)

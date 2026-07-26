from .chroma_tool import search_customer_reviews
from .neo4j_tool import query_purchase_graph
from .postgres_tool import query_postgres_inventory
from .qdrant_tool import search_product_catalog
from .opensearch_tool import full_text_search_docs
from .commit_short_term_memory import commit_short_term_memory


def default_tools():
    """Returns one LangChain tool per database in the docker compose stack."""
    return [
        query_postgres_inventory,
        query_purchase_graph,
        search_customer_reviews,
        search_product_catalog,
        full_text_search_docs,
        commit_short_term_memory,
    ]

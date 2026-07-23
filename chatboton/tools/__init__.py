from .chroma_tool import ChromaTool
from .neo4j_tool import Neo4jTool
from .postgres_tool import PostgresTool
from .qdrant_tool import QdrantTool
from .opensearch_tool import OpenSearchTool


def default_tools():
    """Returns one LangChain tool per database in the docker compose stack."""
    return [
        PostgresTool().as_tool(),
        Neo4jTool().as_tool(),
        ChromaTool().as_tool(),
        QdrantTool().as_tool(),
        OpenSearchTool().as_tool(),
    ]

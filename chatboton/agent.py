"""LangChain tool-calling agent wired to the three database tools."""

from langchain.agents import create_agent

from .providers import get_provider
from .tools import default_tools

SYSTEM_PROMPT = (
    "You are Chatboton, a local assistant for testing AI tool development.\n"
    "You chat normally, and you have several database tools over the same demo "
    "gadget store:\n"
    "- query_postgres_inventory: SQL (SELECT, INSERT, UPDATE) over products(id, name, category, price_eur, stock).\n"
    "- query_purchase_graph: Cypher (MATCH, CREATE, MERGE) over (:Customer)-[:BOUGHT {qty}]->(:Product).\n"
    "- search_customer_reviews: semantic search over customer review texts.\n"
    "- search_product_catalog: vector similarity search in Qdrant.\n"
    "- full_text_search_docs: keyword search in OpenSearch.\n"
    "Product names are shared across all three databases, so you can join "
    "answers (e.g. find a product in reviews, then look up its price in SQL).\n"
    "Use a tool whenever the user asks about products, prices, stock, "
    "customers, purchases, or reviews. Report tool errors honestly.\n"
    "Do not call tools for ordinary conversation."
)


def create_chatboton_agent(provider=None, tools=None):
    """Builds the chat agent.

    Args:
        provider: Optional ``BaseProvider``; defaults to the env-selected
            provider (Ollama with the heretic model unless overridden).
        tools: Optional tool list; defaults to one tool per database.

    Returns:
        Runnable: LangChain agent invocable with ``{"messages": [...]}``.
    """
    provider = provider or get_provider()
    return create_agent(
        model=provider.create_model(),
        tools=tools if tools is not None else default_tools(),
        system_prompt=SYSTEM_PROMPT,
    )

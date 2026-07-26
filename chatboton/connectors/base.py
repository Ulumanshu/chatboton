from abc import ABC, abstractmethod
from typing import Any

class Connector(ABC):
    """Abstract base class for database connectors."""

    @abstractmethod
    def query(self, *args: Any, **kwargs: Any) -> Any:
        """Executes a query against the database."""
        pass

    @abstractmethod
    def insert(self, *args: Any, **kwargs: Any) -> Any:
        """Inserts data into the database."""
        pass

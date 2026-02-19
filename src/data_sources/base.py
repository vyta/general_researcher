"""
Data source interface for retrieving information from public APIs.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetrievedDocument:
    """Represents a document retrieved from a source."""
    source: str
    title: str
    content: str
    url: str
    date: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "date": self.date.isoformat(),
            "metadata": self.metadata
        }


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> "DataSourceResult":
        """Search the data source for relevant documents."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the data source."""
        pass


@dataclass
class DataSourceResult:
    """Result container distinguishing errors from empty results."""
    documents: List[RetrievedDocument]
    error: Optional[str] = None

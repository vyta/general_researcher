"""
Federal Register API data source.
"""
import requests
from typing import List
from datetime import datetime
from .base import DataSource, RetrievedDocument, DataSourceResult


class FederalRegisterDataSource(DataSource):
    """Data source for Federal Register API."""
    
    BASE_URL = "https://www.federalregister.gov/api/v1"
    
    @property
    def name(self) -> str:
        return "Federal Register"
    
    def search(self, query: str, max_results: int = 10) -> DataSourceResult:
        """Search Federal Register documents."""
        documents = []
        
        try:
            url = f"{self.BASE_URL}/documents.json"
            per_page = min(max_results, 1000)
            page = 1
            
            while len(documents) < max_results:
                params = {
                    "conditions[term]": query,
                    "per_page": per_page,
                    "order": "relevance",
                    "page": page
                }
                
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    return DataSourceResult(documents, error=f"{self.name} API error {response.status_code}: {response.text[:200]}")
                
                data = response.json()
                results = data.get("results", [])
                if not results:
                    break
                
                for item in results:
                    if len(documents) >= max_results:
                        break
                    pub_date = item.get("publication_date")
                    date = datetime.strptime(pub_date, "%Y-%m-%d") if pub_date else datetime.now()
                    
                    doc_type = item.get("type", "Unknown")
                    abstract = item.get("abstract", "")
                    agencies_list = item.get("agencies", [])
                    agencies = ", ".join([a.get("name", "") for a in agencies_list])
                    topics = ", ".join(item.get("topics", []))
                    
                    content_parts = [f"Type: {doc_type}"]
                    if abstract:
                        content_parts.append(f"Abstract: {abstract}")
                    if agencies:
                        content_parts.append(f"Agencies: {agencies}")
                    if topics:
                        content_parts.append(f"Topics: {topics}")
                    content = "\n".join(content_parts)
                    
                    doc = RetrievedDocument(
                        source=self.name,
                        title=item.get("title", "Untitled"),
                        content=content,
                        url=item.get("html_url", ""),
                        date=date,
                        metadata={
                            "document_number": item.get("document_number", ""),
                            "type": doc_type,
                            "agencies": agencies_list,
                            "topics": item.get("topics", []),
                            "abstract": abstract
                        }
                    )
                    documents.append(doc)
                
                page += 1
        except Exception as e:
            return DataSourceResult(documents, error=str(e))
        
        return DataSourceResult(documents)

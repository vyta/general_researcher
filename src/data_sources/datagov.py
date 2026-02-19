"""
Data.gov API data source.
"""
import requests
from typing import List
from datetime import datetime
from .base import DataSource, RetrievedDocument, DataSourceResult


class DataGovDataSource(DataSource):
    """Data source for Data.gov catalog."""
    
    BASE_URL = "https://catalog.data.gov/api/3/action"
    
    @property
    def name(self) -> str:
        return "Data.gov"
    
    def search(self, query: str, max_results: int = 10) -> DataSourceResult:
        """Search Data.gov datasets."""
        documents: List[RetrievedDocument] = []
        
        try:
            url = f"{self.BASE_URL}/package_search"
            rows = min(max_results, 1000)
            start = 0
            
            while len(documents) < max_results:
                params = {
                    "q": query,
                    "rows": rows,
                    "start": start
                }
                
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    return DataSourceResult(documents, error=f"{self.name} API error {response.status_code}: {response.text[:200]}")
                
                data = response.json()
                
                if not data.get("success"):
                    return DataSourceResult(documents, error=f"{self.name} API returned success=false")
                
                results = data.get("result", {}).get("results", [])
                if not results:
                    break
                
                for item in results:
                    if len(documents) >= max_results:
                        break
                    metadata_modified = item.get("metadata_modified")
                    date = datetime.fromisoformat(metadata_modified.replace("Z", "+00:00")) if metadata_modified else datetime.now()
                    
                    notes = item.get("notes", "")
                    organization = item.get("organization", {})
                    org_title = organization.get("title", "")
                    tags_list = item.get("tags", [])
                    tags = ", ".join([t.get("display_name", "") for t in tags_list])
                    
                    content_parts = []
                    if notes:
                        content_parts.append(f"Description: {notes}")
                    if org_title:
                        content_parts.append(f"Organization: {org_title}")
                    if tags:
                        content_parts.append(f"Tags: {tags}")
                    content = "\n".join(content_parts) if content_parts else "Dataset metadata unavailable."
                    
                    doc = RetrievedDocument(
                        source=self.name,
                        title=item.get("title", "Untitled Dataset"),
                        content=content,
                        url=f"https://catalog.data.gov/dataset/{item.get('name', '')}",
                        date=date,
                        metadata={
                            "dataset_id": item.get("id", ""),
                            "organization": organization,
                            "tags": tags_list
                        }
                    )
                    documents.append(doc)
                
                start += rows
        except Exception as e:
            return DataSourceResult(documents, error=str(e))
        
        return DataSourceResult(documents)

"""
GovInfo API data source for Congressional bills and legislation.

Uses GovInfo's Search Service API which supports full-text search across
all Congressional bills (collection:BILLS). This is more reliable than
Congress.gov API which has limited search capabilities.

Docs: https://api.govinfo.gov/docs/
"""
import os
import re
import requests
from typing import List, Dict, Any, Tuple
from datetime import datetime
from .base import DataSource, RetrievedDocument, DataSourceResult


class CongressDataSource(DataSource):
    """Data source for Congressional legislation via GovInfo Search API."""
    
    SEARCH_URL = "https://api.govinfo.gov/search"
    
    def __init__(self, api_key: str = None):
        """Initialize with optional API key (defaults to DEMO_KEY)."""
        self.api_key = api_key or os.getenv("GOVINFO_API_KEY") or os.getenv("GOV_API_KEY", "DEMO_KEY")
    
    @property
    def name(self) -> str:
        return "GovInfo"
    
    def search(self, query: str, max_results: int = 10) -> DataSourceResult:
        """
        Search for bills matching the query using GovInfo Search API.
        Supports pagination to retrieve up to max_results documents.
        """
        documents = []
        offset_mark = "*"  # Initial offset mark
        
        try:
            # Build GovInfo search query - use keyword search (not exact phrase)
            search_query = f'{query} collection:BILLS'
            
            while len(documents) < max_results:
                page_size = min(max_results - len(documents), 100)
                
                payload = {
                    "query": search_query,
                    "pageSize": page_size,
                    "offsetMark": offset_mark,
                    "sorts": [
                        {"field": "relevancy", "sortOrder": "DESC"}
                    ]
                }
                
                url = f"{self.SEARCH_URL}?api_key={self.api_key}"
                
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code != 200:
                    return DataSourceResult(documents, error=f"GovInfo API error {response.status_code}: {response.text[:200]}")
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                for result in results:
                    if len(documents) >= max_results:
                        break
                        
                    title = result.get("title", "")
                    package_id = result.get("packageId", "")
                    date_issued = result.get("dateIssued")
                
                    # Parse date
                    if date_issued:
                        try:
                            date = datetime.strptime(date_issued, "%Y-%m-%d")
                        except ValueError:
                            date = datetime.now()
                    else:
                        date = datetime.now()
                    
                    # Parse bill info from packageId (e.g., BILLS-118hr1234ih)
                    bill_type = ""
                    bill_number = ""
                    congress = ""
                    if package_id.startswith("BILLS-"):
                        parts = package_id.replace("BILLS-", "")
                        if len(parts) >= 3:
                            congress = parts[:3]
                            remainder = parts[3:]
                            for btype in ["hjres", "sjres", "hres", "sres", "hr", "s"]:
                                if remainder.startswith(btype):
                                    bill_type = btype
                                    num_part = remainder[len(btype):]
                                    bill_number = re.match(r'\d+', num_part)
                                    bill_number = bill_number.group() if bill_number else ""
                                    break
                    
                    # Build Congress.gov URL for user-friendly viewing
                    if bill_type and bill_number and congress:
                        doc_url = f"https://www.congress.gov/bill/{congress}th-congress/{bill_type}/{bill_number}"
                    else:
                        doc_url = result.get("resultLink", "")
                    
                    summary = result.get("summary", "")
                    subjects = result.get("subjects", [])
                    
                    content = f"Bill: {title}"
                    if summary:
                        content += f"\nSummary: {summary}"
                    if subjects:
                        content += f"\nSubjects: {', '.join(subjects[:5])}"
                    if result.get("download", {}).get("txtLink"):
                        content += f"\nFull text available at GovInfo."
                    
                    authors = result.get("governmentAuthor", [])
                    if authors:
                        content += f"\nAuthors: {', '.join(authors)}"
                    
                    doc = RetrievedDocument(
                        source=self.name,
                        title=title,
                        content=content,
                        url=doc_url,
                        date=date,
                        metadata={
                            "package_id": package_id,
                            "bill_type": bill_type,
                            "bill_number": bill_number,
                            "congress": congress,
                            "collection": result.get("collectionCode", ""),
                        }
                    )
                    documents.append(doc)
                
                # Get next page offset mark
                next_offset = data.get("nextOffsetMark")
                if not next_offset or next_offset == offset_mark:
                    break
                offset_mark = next_offset
                        
        except Exception as e:
            return DataSourceResult(documents, error=str(e))
        
        return DataSourceResult(documents)

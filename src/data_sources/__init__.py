"""
Data source registry and utilities.
"""
import os
from typing import List
from .base import DataSource
from .congress import CongressDataSource
from .federal_register import FederalRegisterDataSource
from .datagov import DataGovDataSource


def get_all_sources(govinfo_api_key: str = None) -> List[DataSource]:
    """Get all available data sources."""
    api_key = govinfo_api_key or os.getenv("GOV_API_KEY")
    return [
        CongressDataSource(api_key=api_key),
        FederalRegisterDataSource(),
        DataGovDataSource(),
    ]

"""
Test data sources individually.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_sources import get_all_sources


def test_sources():
    load_dotenv()
    
    sources = get_all_sources()
    test_query = "artificial intelligence"
    
    print(f"Testing data sources with query: '{test_query}'")
    print("=" * 80)
    
    for source in sources:
        print(f"\n🔍 Testing: {source.name}")
        try:
            result = source.search(test_query, max_results=3)
            if result.error:
                print(f"⚠️ Error: {result.error}")
            print(f"✓ Retrieved {len(result.documents)} documents")
            
            for i, doc in enumerate(result.documents, 1):
                print(f"\n  [{i}] {doc.title}")
                print(f"      Date: {doc.date.strftime('%Y-%m-%d')}")
                print(f"      URL: {doc.url}")
                print(f"      Content preview: {doc.content[:150]}...")
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_sources()

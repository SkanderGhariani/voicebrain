"""Quick local test for memory.search — run: python test_search.py"""

import logging

logging.basicConfig(level=logging.ERROR)

from memory import search

for query in ["restaurant italien", "car problems", "سينما"]:
    print(f"\nQuery: {query}")
    for h in search(query, 3):
        print(f"  #{h['id']} ({h['score']:.2f}) — {h['summary']}")

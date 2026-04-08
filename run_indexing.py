import os
from dotenv import load_dotenv

load_dotenv()

from src.index import index_document

print("Starting indexing pipeline...")
try:
    doc_store = index_document('data/documents/contract.pdf', 'storage/vector_store')
    print("Indexing Success!")
except Exception as e:
    print(f"FAILED indexing: {e}")

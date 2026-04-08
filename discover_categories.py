import os
from dotenv import load_dotenv

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from src.category_discovery import run_discovery

load_dotenv()

def main():
    print("Connecting to existing Chroma Document Store...")
    doc_store = ChromaDocumentStore(persist_path="storage/vector_store")
    
    print("Initiating Category Discovery Pipeline...")
    categories = run_discovery(doc_store)
    
    print("\n================ FINAL INVOICING CATEGORIES ================\n")
    for cat in categories:
        print(f"- {cat}")
    print("\n============================================================\n")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY inside the .env file.")
    else:
        main()

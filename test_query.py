import os
from dotenv import load_dotenv

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from src.pipeline import build_retrieval_pipeline, query_system

# Load environment variables
load_dotenv()

def test_query():
    print("Loading existing Chroma Document Store...")
    doc_store = ChromaDocumentStore(persist_path="storage/vector_store")
    
    print("Building Retrieval Pipeline...")
    query_pipe = build_retrieval_pipeline(doc_store)
    
    question = "What are the exact conditions under which this agreement can be terminated? Is there a notice period?"
    print(f"\n[Question] {question}\n")
    
    print("Querying system (this may take a few seconds)...")
    answer = query_system(query_pipe, question)
    
    print("\n================== ANSWER ==================")
    print(answer)
    print("============================================")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Please make sure your OPENAI_API_KEY is set in the .env file!")
    else:
        test_query()

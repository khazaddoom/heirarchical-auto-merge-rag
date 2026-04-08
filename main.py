import os
from dotenv import load_dotenv

# Load environment variables (such as OPENAI_API_KEY) from .env file
load_dotenv()

from src.index import index_document
from src.pipeline import build_retrieval_pipeline, query_system

def main():
    # 1. Ensure API key is present
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set in the environment or .env file.")
        print("Please add your key to a .env file: OPENAI_API_KEY=sk-...")
        return
        
    pdf_path = "data/documents/contract.pdf"
    
    # 2. Run Indexing / load the active DocumentStore
    print("\n--- Phase 1: Indexing ---")
    try:
        doc_store = index_document(pdf_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please place a readable PDF named 'contract.pdf' in the data/documents/ folder.")
        return
        
    # 3. Build Query Pipeline
    print("\n--- Phase 2: Building Query Pipeline ---")
    query_pipe = build_retrieval_pipeline(doc_store)
    
    # 4. Interactive loop answering
    print("\n--- Phase 3: Ready for Queries ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        question = input("\nAsk a question about the document: ")
        if question.lower().strip() in ["exit", "quit", "q"]:
            break
            
        if not question.strip():
            continue
            
        print("Thinking...")
        answer = query_system(query_pipe, question)
        print(f"\n=> Answer: {answer}")

if __name__ == "__main__":
    main()

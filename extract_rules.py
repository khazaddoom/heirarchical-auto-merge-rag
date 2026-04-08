import os
import json
from dotenv import load_dotenv

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from src.category_discovery import run_discovery
from src.rule_extractor import build_extraction_pipeline, extract_rules_for_category

load_dotenv()

def main():
    print("====================================")
    print(" Initiating Auto-Rule Engine V1 ")
    print("====================================\n")
    
    print("Connecting to local Chroma document store...")
    doc_store = ChromaDocumentStore(persist_path="storage/vector_store")
    
    print("\n[PHASE 1] Autonomously discovering invoicing categories...")
    categories = run_discovery(doc_store)
    
    print(f"\nDiscovered {len(categories)} distinct commercial themes.")
    for c in categories:
        print(f"- {c}")
        
    print("\n[PHASE 2] Executing Agentic Multi-pass Rule RAG...")
    extraction_pipe = build_extraction_pipeline(doc_store)
    
    compiled_rulebook = []
    
    for cat in categories:
        # Pass each category to the RAG LLM dynamically
        rules = extract_rules_for_category(extraction_pipe, cat)
        if rules:
            compiled_rulebook.extend(rules)
            print(f"      [✓] Extracted {len(rules)} specific structured rules")
            
    print("\n[PHASE 3] Compiling and persisting Rulebook...")
    
    # Save the consolidated JSON file
    os.makedirs("data", exist_ok=True)
    out_path_json = "data/rulebook.json"
    with open(out_path_json, "w") as f:
        json.dump({"total_rules": len(compiled_rulebook), "rules": compiled_rulebook}, f, indent=4)
        
    print(f"      [✓] Saved structured JSON schema to {out_path_json}")

    # Generate Human-Readable Excel Dump
    import pandas as pd
    if compiled_rulebook:
        df = pd.DataFrame(compiled_rulebook)
        out_path_excel = "data/rulebook.xlsx"
        df.to_excel(out_path_excel, index=False)
        print(f"      [✓] Saved human-readable tabular dump to {out_path_excel}")
        
    print(f"\n[DONE] {len(compiled_rulebook)} total rules successfully generated!")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY inside the .env file.")
    else:
        main()

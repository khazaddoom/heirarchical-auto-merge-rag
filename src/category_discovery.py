import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.types import DocumentStore

MAP_TEMPLATE = """
You are a highly analytical contract scanning agent.
Analyze the following section of a contract and identify any distinct subject categories involving financial terms, invoicing, billing, pricing, markups, or commercial limits.
If there are no such commercial categories in this specific text, output exactly "NONE". 
Otherwise, output a highly concise bulleted list of the high-level category names (e.g. "Overtime Rates", "Materials Surcharge"). Do not output the rules or values themselves, only the category headers.

Contract Text:
----------
{{ text }}
----------

Categories:
"""

REDUCE_TEMPLATE = """
You are a commercial data architect streamlining invoice rules.
Below is a raw, overlapping list of commercial and billing topics extracted from various independent sections of a massive contract.

Raw Distinct Categories Extracted:
----------
{{ raw_categories }}
----------

Your task is to consolidate this massive list into a clean, canonical array of roughly 5 to 12 distinct, high-level invoice categories (e.g., "Hourly Labor Rates", "Material Markups", "Travel & Expense Policies", "Overtime & Weekend Policies").
Output these final consolidated categories strictly as a bulleted list with no extra preamble or conversation.
"""

class CategoryDiscoveryEngine:
    def __init__(self, model: str = "gpt-4o-mini"):
        # Initialize generic LLM Generators
        self.map_llm = OpenAIGenerator(model=model)
        self.reduce_llm = OpenAIGenerator(model=model)
        
        # Build Map Pipeline
        self.map_pipe = Pipeline()
        self.map_pipe.add_component("prompt", PromptBuilder(template=MAP_TEMPLATE))
        self.map_pipe.add_component("llm", self.map_llm)
        self.map_pipe.connect("prompt", "llm")
        
        # Build Reduce Pipeline
        self.reduce_pipe = Pipeline()
        self.reduce_pipe.add_component("prompt", PromptBuilder(template=REDUCE_TEMPLATE))
        self.reduce_pipe.add_component("llm", self.reduce_llm)
        self.reduce_pipe.connect("prompt", "llm")

    def _process_chunk_map(self, text: str) -> str:
        """Processes a single chunk through the Map pipeline."""
        res = self.map_pipe.run({"prompt": {"text": text}})
        reply = res["llm"]["replies"][0].strip()
        if reply.upper() == "NONE" or "NONE" in reply[:10].upper():
            return ""
        return reply

    def discover_categories(self, doc_store: DocumentStore) -> List[str]:
        """
        Executes a Map-Reduce category discovery over all large sections of the indexed contract.
        """
        print("Fetching Level 1 Root Sections for discovery...")
        # Try to explicitly pull only 'level 1' structural nodes for broad scanning.
        # If the store is empty, this returns empty.
        docs = doc_store.filter_documents(filters={"operator": "==", "field": "__level", "value": 1})
        
        # Failsafe if filter didn't work or document had no levels (e.g. mock stores)
        if not docs:
            print("No Level 1 docs found by filter, pulling all documents as a fallback for discovery...")
            docs = [d for d in doc_store.filter_documents() if d.content]
            
        print(f"Executing broad context Map scan over {len(docs)} large chunks in parallel...")
        
        raw_category_lists = []
        
        # We use ThreadPoolExecutor to run the LLM calls concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all map operations
            futures = [executor.submit(self._process_chunk_map, doc.content) for doc in docs]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    raw_category_lists.append(result)
                    
        # Combine the map outputs
        combined_text = "\n\n".join(raw_category_lists)
        if not combined_text.strip():
            return ["No commercial or invoicing categories were detected in the provided documents."]
            
        print("\nExecuting Reduce scan to consolidate themes...")
        res = self.reduce_pipe.run({"prompt": {"raw_categories": combined_text}})
        final_list = res["llm"]["replies"][0].strip()
        
        # Parse into a python list by splitting on newlines/bullets
        clean_categories = []
        for line in final_list.split('\n'):
            line = line.strip()
            # Remove markdown bullets, asterisks, dashes, numbers
            for chars in ['*', '-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']:
                if line.startswith(chars):
                    line = line[len(chars):].strip()
                    
            if line:
                clean_categories.append(line)
                
        return clean_categories

# Main helper access point
def run_discovery(doc_store: DocumentStore) -> List[str]:
    engine = CategoryDiscoveryEngine()
    return engine.discover_categories(doc_store)

import json
from typing import List
from pydantic import ValidationError

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

from src.rule_schema import Rulebook

EXTRACT_PROMPT = """
You are a precise, automated invoice compliance algorithm.
Your objective is to extract rigidly structured rules pertaining specifically to the commercial category: "{{category}}" from the provided, legally binding contract context.

Contract Context:
-----------------
{% for doc in documents %}
{{ doc.content }}
{% endfor %}
-----------------

CRITICAL INSTRUCTIONS:
1. Extract all specific thresholds, limitations, and financial logic.
2. If the text specifies that actual numeric rates or shift details are defined inside an external attachment, schedule, or Appendix (e.g. "detailed in Appendix E" or "as per attached Excel matrix"), you MUST extract that as a rule, set "requires_external_lookup" to true, and record the exact phrase into "external_reference_source".
3. If no rules for this precise category are found in the context provided, output an empty rules array.

OUTPUT FORMAT:
You must output strictly valid JSON conforming exactly to this schema:
{{schema}}
"""

def build_extraction_pipeline(doc_store: ChromaDocumentStore) -> Pipeline:
    """Builds the Auto-Merging Retrieval pipeline wired for structured JSON extraction."""
    pipe = Pipeline()
    
    embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    
    # Query only the purest Level 3 Leaf Nodes (approx 64 words).
    retriever = ChromaEmbeddingRetriever(
        document_store=doc_store, 
        top_k=20,
        filters={"operator": "==", "field": "__level", "value": 3}
    )
    
    # AutoMerger will recursively snap Level 3 leaves into larger Level 2 or Level 1 contexts 
    # if >50% hit density is detected, providing seamless context!
    merger = AutoMergingRetriever(document_store=doc_store, threshold=0.5)
    
    prompt = PromptBuilder(template=EXTRACT_PROMPT)
    
    # Force the OpenAI generator to strictly return valid JSON structures
    generator = OpenAIGenerator(
        model="gpt-4o-mini",
        generation_kwargs={"response_format": {"type": "json_object"}}
    )
    
    pipe.add_component("text_embedder", embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("merger", merger)
    pipe.add_component("prompt_builder", prompt)
    pipe.add_component("llm", generator)
    
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever", "merger")
    pipe.connect("merger.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")
    
    return pipe

def extract_rules_for_category(pipe: Pipeline, category: str) -> List[dict]:
    """Execute the multi-pass query per category and parse the JSON response safely."""
    # We dynamically pass the Pydantic schema json schema directly into the prompt!
    schema_str = json.dumps(Rulebook.model_json_schema(), indent=2)
    
    print(f"  [>] Agentically scanning vector space and orchestrating RAG context for: {category}...")
    # By asking specifically about the category, we embed the concept and match closest chunks
    query = f"Extract all rules, terms, or conditions regarding: {category}"
    
    try:
        res = pipe.run({
            "text_embedder": {"text": query},
            "prompt_builder": {"category": category, "schema": schema_str}
        })
        
        reply_json_str = res["llm"]["replies"][0]
        parsed_dict = json.loads(reply_json_str)
        # Validate structurally to guarantee alignment before trusting it!
        rulebook_model = Rulebook(**parsed_dict)
        
        # We output standard dicts internally to merge lists easily
        return [r.model_dump() for r in rulebook_model.rules]
        
    except ValidationError as ve:
        print(f"      [!] Schema validation error occurred for {category}: {ve}")
        return []
    except json.JSONDecodeError:
        print(f"      [!] Broken JSON response returned by LLM for {category}")
        return []
    except Exception as e:
        print(f"      [!] Pipeline failure on {category}: {e}")
        return []

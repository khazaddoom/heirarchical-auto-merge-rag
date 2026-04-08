# Milestone 2: Agentic Rule Extraction & Categorization System

## Overview
This milestone establishes a systematic, fully automated pipeline capable of digesting complex, unstructured commercial contract documents and forcefully converting them into an organized list of **strictly programmatic invoicing rules** designed for future algorithmic validation.

## Architectural Phases

### Phase 1: Contextual Category Discovery (`src/category_discovery.py`)
Rather than risking hallucinations by passing a blind "extract all rules" prompt over a massive document, we implemented a sophisticated Map-Reduce discovery engine.
* **Map:** We autonomously pull the highest-level (`Level 1` - 1024 word blocks) chunks from ChromaDB and process them using concurrent AI threads to identify deeply localized themes.
* **Reduce:** We summarize all localized themes into a canonical list of ~5-15 high-level commercial rulesets (e.g., *Compensation Structures*, *Material Markups*).

### Phase 2: Agentic Multi-Pass RAG Extraction (`src/rule_extractor.py`)
To achieve enterprise reliability, the engine uses the discovered categories to orchestrate multi-pass retrieval sessions.
* **Targeted context:** Using the `AutoMergingRetriever`, the system loops over each discovered category, pulling highly-focused, granular contexts directly correlating to the specific category at hand.
* **Pydantic Schema Strictness:** The system natively forces the AI to output strictly structured JSON arrays reflecting the constraints established in `src/rule_schema.py`.

## Data Outputs
The resulting Extractions are dual-persisted locally via `pandas` execution during pipeline wrap-up.
1. **`data/rulebook.json`:** The raw programmatic tree, ready for ingestion into Python validation code.
2. **`data/rulebook.xlsx`:** A strictly formatted, human-readable tabular ledger perfect for manual auditing.

## Critical Engine Features Implemented
* **Dodge Hallucinations via Auto-Mergers:** By requesting exact terms exclusively inside context, hallucinations are near zero.
* **External Reference Flagging (`requires_external_lookup`):** We successfully engineered the LLM prompt to actively analyze the semantics of sentences requiring external documentation (e.g. Scaffolding Material costs tied to 'Exhibit CR 6') and set up a boolean flag so the future validator knows it must ingest secondary tables!

---

## Operating Instructions (Rebuilding the System)

If you need to tweak your chunk sizes to experiment with how context windows affect extracted rules, follow this sequence precisely:

### 1. Tweak the Chunk Metrics
Open up `src/ingest.py` and modify **line 17**:
```python
block_sizes = [1024, 256, 64]
```

### 2. Wipe the Chroma Cache
Because the mathematical relationships of the embeddings change, you **must** entirely purge the old database directory.
Run this from your terminal:
```bash
rm -rf storage/vector_store
```

### 3. Re-Deploy the System
Run the scripts sequentially to freshly index the contract and fire the dual discovery/extraction pipelines:
```bash
source .venv/bin/activate

# Execute Indexing Engine
python run_indexing.py

# Execute AI Discovery & RAG Extractions
python extract_rules.py
```
This final step will overwrite the `data/rulebook` JSON and Excel payloads with the latest architecture.

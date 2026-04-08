# Milestone 1: Hierarchical Auto-Merging RAG Implementation

This document serves as a complete history and technical breakdown of our initial build phase, summarizing the architectural decisions, pipeline construction, and the specific debugging challenges we overcame to reach stable execution.

---

## 1. Architectural Decisions

1. **Library Pivot**: We initially reviewed instructions targeting `llama-index`, but elected to pivot entirely to the **Haystack 2.0** framework. Haystack provided highly explicit internal components (`HierarchicalDocumentSplitter`, `AutoMergingRetriever`) mapping beautifully into customizable Pipelines.
2. **Vector Store**: We selected **ChromaDB** (`ChromaDocumentStore`) for persistent vectorized node storage, as it elegantly respects custom metadata (parent/child links) natively.
3. **LLM & Embeddings**: We opted for **OpenAI** across the board. We integrated `text-embedding-3-small` (1536 dim) for granular chunk embedding, and `gpt-4o-mini` for the final text generation constraint.

## 2. Pipeline Construction

The implementation was split into discrete, decoupled Haystack pipelines to allow isolated validation and testing:

* **Ingestion (`src/ingest.py`)**: Utilized `PyPDFToDocument` to parse raw PDF contracts. Pushed data into the `HierarchicalDocumentSplitter` to create three distinct tier resolutions: large section context (1024 words), intermediate (256 words), and precision leaf search (64 words).
* **Indexing (`src/index.py`)**: Tied ingestion into `OpenAIDocumentEmbedder` and stored final node graphs in `storage/vector_store`.
* **Querying (`src/pipeline.py`)**: Embedded the user query, retrieved the top 15 closest leaf nodes, fired the `AutoMergingRetriever` to seamlessly merge sibling nodes backwards into deeper contextual parent nodes (upon a 50% threshold hit), and dispatched the result via a custom `qa_prompt_template`.
* **Entrypoints**: Wrote `main.py` for interactive chatting and `test_query.py` for headless unit testing.

## 3. Debugging Log & Fixes

We sequentially encountered and solved several complex issues resulting from Haystack's internal architectures and unstructured PDF constraints:

### A. The NLTK Tokenizer Wall
* **Error**: Haystack violently rejecting `split_by="sentence"` during testing (`ImportError: No module named 'nltk'`).
* **Fix**: Force-installed the backend `nltk` libraries and explicitly downloaded the `punkt` and `punkt_tab` natural language datasets.

### B. The 8192 Token API Crash
* **Error**: OpenAI rejected indexing via: `Invalid 'input[0]': maximum input length is 8192 tokens`.
* **Root Cause**: The PDF (`contract.pdf`) lacked proper sentence punctuation format. Because we were splitting by "sentences", the splitter found massive 64,000-word run-on blocks and passed them natively to OpenAI.
* **Fix**: Reverted the chunking mechanic entirely to `split_by="word"` and initialized mathematically safe caps `[1024, 256, 64]`. At ~1.3 tokens per word, this guaranteed we would mathematically never breach an 8192 context window.

### C. The Level 0 "Original Document" Sinkhole
* **Error**: OpenAI immediately crashed again upon embedding.
* **Root Cause**: Haystack's `HierarchicalDocumentSplitter` yields *both* the perfectly sized chunks *AND* the raw, unaltered original source 433k-character document secretly flagged as `__level: 0`.
* **Fix**: We engineered a custom Haystack Component interceptor (`DocumentLevelFilter` in `src/index.py`). This strictly purges any Node mapped to `__level: 0` before it hits the Embedder API!
* **Secondary Typo Fix**: Fixed a strict Haystack typing pipeline validator crash by converting Python's loose `list` back to strict `List[Document]` decorators in the newly minted `DocumentLevelFilter`.

### D. Chroma Dimension Conflict
* **Error**: `chromadb.errors.InvalidArgumentError: Collection expecting embedding with dimension of 384, got 1536`
* **Fix**: The legacy dummy database initialized locally tracking MiniLM sizes (384) dynamically rejected the new OpenAI streams (1536). Forcefully executing `rm -rf storage/vector_store` reset the Chroma constraints completely.

### E. The AutoMerger Parent Disconnect Crash
* **Error**: The retrieval pipeline panicked loudly finding no parent keys: `ValueError: Expected 1 parent document with id [...], found 0`.
* **Root Cause**: Two logical flaws stacked on top of each other:
    1. The `ChromaEmbeddingRetriever` retrieved at all depths blindly, pulling Level 1 (Top Level) parent chunks.
    2. Because it was pulled via retrieval, `AutoMergingRetriever` naturally attempted to search for Level 1's upstream metadata parent (`Level 0`), which we intentionally filtered out to save API quota!
* **Fix**: Applied a highly specific query retrieval filter directly inside `src/pipeline.py`: `filters={"operator": "==", "field": "__level", "value": 3}`. This completely banned the retriever from searching anything but granular Level 3 leaves. Furthermore, we explicitly set `__parent_id` to `None` for Level 1 nodes mapping them properly as absolute Roots.

## 4. Final Environment Commands Executed

```bash
# Building the isolated Python ecosystem to bypass Mac OS system limits
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3.11 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab');"

# Hard Resetting Database
rm -rf storage/

# Generating the Vector Embeddings
python run_indexing.py

# Running the Tests and Final UI
python test_query.py
python main.py
```

## Summary Status
The backend architecture is **STABLE**.
All components are fully validated, logically synced, and pushed securely to Git with robust `.gitignore` protections for all API tokens.

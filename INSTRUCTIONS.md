# Auto-Merging RAG — Implementation Plan

A general implementation plan for building a Retrieval-Augmented Generation (RAG) application using the **Auto-Merging Retrieval** technique with a hierarchical document structure, applied to PDF documents.

---

## Overview

Auto-Merging Retrieval stores documents in a **hierarchical node tree** (parent → child → leaf). Only the small leaf nodes are embedded and searched, but if enough sibling leaves match a query, they are "merged up" into their parent node — giving the LLM broader, coherent context instead of many scattered fragments.

---

## Phase 1 — Environment Setup

Install the required dependencies:

```bash
pip install llama-index pypdf sentence-transformers openai chromadb
```

Key libraries:

- `llama-index` — has first-class support for `AutoMergingRetriever`
- `pypdf` or `pdfminer.six` — PDF text extraction
- `sentence-transformers` — local embeddings (or use OpenAI's API)
- `chromadb` / `qdrant` / `weaviate` — vector store

---

## Phase 2 — PDF Ingestion & Text Extraction

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["your_document.pdf"]
).load_data()
```

Considerations:

- Use `pypdf` for text-native PDFs; use `unstructured` or `pdfplumber` for scanned/OCR-needed PDFs.
- Preserve page-level metadata (page number, section heading) for later citation.
- Clean extracted text: strip headers/footers, fix hyphenation across lines.

---

## Phase 3 — Hierarchical Node Construction

This is the core of auto-merging. Define **multiple chunk sizes** in a parent-to-child ratio:

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]   # L1 → L2 → L3
)

all_nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(all_nodes)
```

How the hierarchy works:

| Level | Chunk Size | Role |
|-------|------------|------|
| L1 — Root | ~2048 tokens | Full section / chapter context |
| L2 — Mid | ~512 tokens | Paragraph-level context |
| L3 — Leaf | ~128 tokens | Fine-grained search unit (indexed) |

Only **leaf nodes are embedded**. Parent nodes are stored in a `docstore` and referenced by ID.

---

## Phase 4 — Indexing (Storage)

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

# Store ALL nodes (parent + leaf) in the docstore
docstore = SimpleDocumentStore()
docstore.add_documents(all_nodes)

# Index ONLY leaf nodes for vector search
storage_context = StorageContext.from_defaults(docstore=docstore)
index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)
```

Why separate stores: the vector store only needs leaf embeddings for fast retrieval. The docstore holds the full hierarchy so the merge step can "walk up" the tree to fetch parent nodes without re-embedding them.

---

## Phase 5 — Auto-Merging Retriever

```python
from llama_index.retrievers import AutoMergingRetriever

base_retriever = index.as_retriever(similarity_top_k=12)

retriever = AutoMergingRetriever(
    vector_retriever=base_retriever,
    storage_context=storage_context,
    simple_ratio_thresh=0.5,   # merge if ≥50% of a parent's children are retrieved
    verbose=True
)
```

The merge decision logic:

```
For each retrieved leaf node:
  → Find its parent node
  → Count how many of the parent's children were also retrieved
  → If (retrieved_children / total_children) ≥ threshold:
        replace all those children with the single parent node
  → Else:
        keep the individual leaf nodes
```

This prevents the LLM from seeing the same information fragmented across small chunks, reducing noise and improving coherence.

---

## Phase 6 — Query Engine & Response Generation

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

synthesizer = get_response_synthesizer(response_mode="compact")

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=synthesizer,
)

response = query_engine.query("What are the key findings in section 3?")
print(response)
```

---

## Phase 7 — Evaluation & Tuning

Metrics to track:

| Metric | Tool | What it measures |
|--------|------|------------------|
| Context precision | `ragas` | Are retrieved chunks actually relevant? |
| Context recall | `ragas` | Did we miss important chunks? |
| Answer faithfulness | `ragas` | Is the answer grounded in context? |
| Merge rate | custom | % of queries triggering a merge |

Key hyperparameters to tune:

- `chunk_sizes` — try `[1024, 256, 64]` for denser documents
- `similarity_top_k` — higher gives more merge opportunities but increases cost
- `simple_ratio_thresh` — lower threshold = more aggressive merging

---

## Phase 8 — Recommended Project Structure

```
rag_app/
├── data/
│   └── documents/         # PDF files go here
├── storage/
│   ├── docstore.json      # persisted node hierarchy
│   └── vector_store/      # persisted embeddings
├── src/
│   ├── ingest.py          # PDF parsing + hierarchical chunking
│   ├── index.py           # build & persist the index
│   ├── retriever.py       # auto-merging retriever setup
│   └── query.py           # query engine + response
├── eval/
│   └── evaluate.py        # ragas evaluation scripts
└── main.py
```

---

## Key Design Decisions

- **Chunk ratio:** Keep the parent roughly 4× the size of its child — this gives enough "overlap signal" for the merge threshold to fire meaningfully.
- **Leaf size:** Smaller leaves (64–128 tokens) improve retrieval precision but increase the number of nodes; tune based on document density.
- **Merge threshold:** Start at `0.4`–`0.5`. Lower it if the LLM lacks context; raise it if responses contain too much irrelevant material.
- **Embedding model:** Use the same model for both indexing and query embedding — mismatched models are a common source of poor retrieval quality.
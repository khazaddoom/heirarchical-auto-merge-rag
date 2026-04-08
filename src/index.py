import os
from pathlib import Path
from typing import List
from haystack import Document, Pipeline, component
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from src.ingest import IngestionPipeline

@component
class DocumentLevelFilter:
    """Drops the massive original (Level 0) document and sets Level 1 to be the new root."""
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        filtered_docs = []
        for doc in documents:
            level = doc.meta.get("__level", 0)
            if level > 0:
                # If it is level 1, disconnect it from the massive level 0 document
                if level == 1:
                    doc.meta["__parent_id"] = None
                filtered_docs.append(doc)
        return {"documents": filtered_docs}

def get_indexing_pipeline(doc_store: ChromaDocumentStore) -> Pipeline:
    """
    Constructs the full indexing pipeline:
    1. PyPDFToDocument converter
    2. HierarchicalDocumentSplitter
    3. DocumentLevelFilter (drops source document)
    4. OpenAIDocumentEmbedder
    5. DocumentWriter
    """
    ingestor = IngestionPipeline()
    level_filter = DocumentLevelFilter()
    embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
    writer = DocumentWriter(document_store=doc_store)
    
    pipe = Pipeline()
    pipe.add_component("converter", ingestor.pdf_converter)
    pipe.add_component("splitter", ingestor.splitter)
    pipe.add_component("filter", level_filter)
    pipe.add_component("embedder", embedder)
    pipe.add_component("writer", writer)
    
    pipe.connect("converter", "splitter")
    pipe.connect("splitter", "filter")
    pipe.connect("filter", "embedder")
    pipe.connect("embedder", "writer")
    
    return pipe

def index_document(pdf_path: str, persist_dir: str = "storage/vector_store"):
    """
    Checks if a document needs processing and indexes it into Chroma.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF missing at {pdf_path}")
        
    print(f"Connecting to Chroma at {persist_dir}...")
    doc_store = ChromaDocumentStore(persist_path=persist_dir)
    
    # If there are already docs, for simplicity we skip indexing 
    # (in a real app, you might do checking by file hash)
    count = doc_store.count_documents()
    if count > 0:
        print(f"Store already contains {count} documents. Skipping indexing to save API tokens.")
        return doc_store
        
    indexing_pipe = get_indexing_pipeline(doc_store)
    
    print(f"Starting hierarchical indexing for {pdf_path} (This uses OpenAI embeddings...)")
    indexing_pipe.run({"converter": {"sources": [pdf_path]}})
    
    print(f"Indexing complete! Store now contains {doc_store.count_documents()} total nodes (parents + children).")
    return doc_store

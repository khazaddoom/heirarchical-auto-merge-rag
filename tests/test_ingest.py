import os
import pytest
from pathlib import Path
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Ensure relative imports work if we run pytest from root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import IngestionPipeline

@pytest.fixture
def sample_pdf(tmp_path):
    """
    Creates a simple blank-ish PDF for testing purposes.
    (Requires pypdf to create, but we can just write minimum valid pdf bytes).
    """
    pdf_path = tmp_path / "test_doc.pdf"
    from pypdf import PdfWriter
    
    # Create a tiny pdf with some text
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Actually pypdf add_blank_page creates an empty page. Let's just create a raw text Document manually for the first test
    # instead of doing PDF generation which is tricky with simple pypdf.
    # We will test the PDF parsing and Splitting in separate steps or use fake docs.
    return str(pdf_path)


def test_hierarchical_splitter_behavior():
    """
    Test Case 1: Verify document splits into correct parent/child hierarchies.
    We test the splitter manually with a large text block.
    """
    # Create an ingestion pipeline with small block sizes for easy testing
    pipeline = IngestionPipeline(block_sizes=[30, 10])
    
    # words text
    text = " ".join([f"word_{i}" for i in range(100)])
    doc = Document(content=text)
    
    # Run the splitter directly
    result = pipeline.splitter.run(documents=[doc])
    chunks = result["documents"]
    
    # Ensure splits happened
    assert len(chunks) > 1, "Documents were not split"
    
    # In hierarchical splitting, parent documents and children are yielded or stored 
    # Check for metadata indicating parent-child linkage.
    # Haystack usually injects `_split_id` or similar metadata.
    # Let's verify we have documents of different sizes (some should be parent ~30 words, some children ~10 words)
    words_lengths = [len(str(c.content).split()) for c in chunks]
    
    has_large_chunks = any(l >= 15 for l in words_lengths) # parent chunk length
    has_small_chunks = any(l <= 10 for l in words_lengths) # child chunk length
    
    assert has_large_chunks, "Missing parent-level large chunks"
    assert has_small_chunks, "Missing child-level small chunks"


def test_ingestion_pipeline_with_in_memory_store():
    """
    Test Case 2: Verify components can be chained in a Haystack Pipeline and write to store.
    Instead of PDF, we mock the pipeline.
    Actually we can just run the pre-built pipeline using a mock DocumentStore.
    """
    from haystack.components.writers import DocumentWriter
    # Initialize pipeline provider
    provider = IngestionPipeline(block_sizes=[20, 5])
    
    # Setup InMemory store
    doc_store = InMemoryDocumentStore()
    
    # We will just replace the pipeline's converter with a passthrough for testing
    from haystack.components.builders import PromptBuilder
    from haystack import Pipeline
    
    pipe = Pipeline()
    pipe.add_component("splitter", provider.splitter)
    pipe.add_component("writer", DocumentWriter(document_store=doc_store))
    pipe.connect("splitter", "writer")
    
    # 10 sentences doc
    test_doc = Document(content=" ".join([f"This is test token sentence {i}." for i in range(10)]))
    
    pipe.run({"splitter": {"documents": [test_doc]}})
    
    stored_docs = doc_store.filter_documents()
    assert len(stored_docs) > 0, "No documents were written to the store"

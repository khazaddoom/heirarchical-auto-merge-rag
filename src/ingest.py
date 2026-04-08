import os
from pathlib import Path
from typing import List

from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import HierarchicalDocumentSplitter
# from haystack_integrations.document_stores.chroma import ChromaDocumentStore
# from haystack.components.writers import DocumentWriter

# Initialize the simple document processing components
class IngestionPipeline:
    def __init__(self, block_sizes: list[int] = None, split_overlap: int = 0):
        """
        Initialize the ingestion pipeline components.
        For hierarchical splitting, we pass a list of block sizes.
        We use word splitting to guarantee chunk limits don't exceed OpenAI's 8192 token max.
        """
        if block_sizes is None:
            block_sizes = [768, 256, 64]
            
        self.block_sizes = block_sizes
        self.pdf_converter = PyPDFToDocument()
        self.splitter = HierarchicalDocumentSplitter(
            block_sizes=self.block_sizes, 
            split_by="word",
            split_overlap=split_overlap
        )
        
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process a single PDF into hierarchical documents.
        This provides a manual sequential approach for testing.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
            
        # Convert PDF to Document
        doc_res = self.pdf_converter.run(sources=[pdf_path])
        documents = doc_res["documents"]
        
        # Split documents hierarchically
        split_res = self.splitter.run(documents=documents)
        return split_res["documents"]
        
    def get_ingestion_pipeline(self, document_store) -> Pipeline:
        """
        Get the full Haystack Pipeline ready to run.
        (We pass `document_store` generically so we can test with InMemoryDocumentStore)
        """
        # We need a DocumentWriter to write into the given store
        from haystack.components.writers import DocumentWriter
        
        writer = DocumentWriter(document_store=document_store)
        
        pipeline = Pipeline()
        pipeline.add_component("converter", self.pdf_converter)
        pipeline.add_component("splitter", self.splitter)
        pipeline.add_component("writer", writer)
        
        pipeline.connect("converter", "splitter")
        pipeline.connect("splitter", "writer")
        
        return pipeline

if __name__ == "__main__":
    # Example manual usage:
    # ingestor = IngestionPipeline()
    # docs = ingestor.process_pdf("data/documents/sample.pdf")
    # print(f"Produced {len(docs)} chunks.")
    pass

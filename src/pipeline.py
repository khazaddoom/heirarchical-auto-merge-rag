from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# A custom prompt that feeds the auto-merged context to the LLM
qa_prompt_template = """
You are an expert assistant analyzing documents via a Retrieval-Augmented Generation system.
Based ONLY on the provided context documents, answer the question.
If the answer is not in the documents, just say "I'm sorry, I cannot answer that based on the provided documents."

Context Documents:
{% for doc in documents %}
--- Document Chunk ---
{{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer:
"""

def build_retrieval_pipeline(doc_store: ChromaDocumentStore) -> Pipeline:
    """
    Constructs the query pipeline:
    1. OpenAITextEmbedder (embeds the user's question)
    2. ChromaEmbeddingRetriever (fetches top matching chunks)
    3. AutoMergingRetriever (merges sibling child chunks into their parent if threshold met)
    4. PromptBuilder (injects merged contexts and query into the LLM prompt)
    5. OpenAIGenerator (the LLM answering the question)
    """
    
    # Initialize all components
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    
    # We must explicitly query ONLY the lowest-level leaf chunks (Level 3 for [1024, 256, 64]).
    # Otherwise it retrieves intermediate nodes and the AutoMerger fails.
    chroma_retriever = ChromaEmbeddingRetriever(
        document_store=doc_store, 
        top_k=15,
        filters={"operator": "==", "field": "__level", "value": 3}
    )
    
    # AutoMergingRetriever sits after the initial retrieval.
    # threshold=0.5 means if 50% or more of a parent's children are retrieved,
    # it completely replaces all children with the one large parent document for superior coherence.
    merger = AutoMergingRetriever(document_store=doc_store, threshold=0.5)
    
    prompt_builder = PromptBuilder(template=qa_prompt_template)
    
    llm = OpenAIGenerator(model="gpt-4o-mini")
    
    # Assemble pipeline
    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", chroma_retriever)
    pipe.add_component("merger", merger)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    
    # Connect
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "merger.documents")
    pipe.connect("merger.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")
    
    return pipe

def query_system(pipeline: Pipeline, question: str):
    """
    Executes a query and prints the response.
    """
    res = pipeline.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"query": question}
    })
    
    answer = res["llm"]["replies"][0]
    return answer

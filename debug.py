from src.ingest import IngestionPipeline
pipe = IngestionPipeline()
docs = pipe.process_pdf("data/documents/contract.pdf")
sizes = [len(doc.content) for doc in docs if doc.content]
sizes.sort(reverse=True)
print(sizes[:10])
print(len(docs))

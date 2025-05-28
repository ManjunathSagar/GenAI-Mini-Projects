import os
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
#from pydantic import Secret
from dotenv import load_dotenv

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = []
for doc in dataset:
    print(f"Doc from HF - {doc} \n")
    docs.append(Document(content=doc["content"], meta=doc["meta"]))
    
# This document store suitable for only small data sets and practice
document_store = InMemoryDocumentStore()

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
#Warming up helps us download the model and get us ready for future use
doc_embedder.warm_up()

#Let's create embeddings for this model and assign all documents to docs
docs_with_embeddings = doc_embedder.run(docs)

for doc in docs_with_embeddings["documents"]:
    print(f"Doc with embedding -{doc.embedding}")

#Let's write this document embeddings to document store
document_store.write_documents(docs_with_embeddings["documents"])
#Adding document embeddings to embedding retriever for future to retrieving documents 
#Initialize inmemory embedding retriever 
retriever = InMemoryEmbeddingRetriever(document_store)

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

template = """
Given the folowing information, answer the question.

Context:
{% for document in documents %} 
    {{ document.content }}

{% endfor %}

Question: {{ question }}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
generator = OpenAIGenerator(model="gpt-3.5-turbo")

basic_rag_pipeline = Pipeline()

#Add all your required components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

#Let us connect these components in the pipeline
basic_rag_pipeline.connect("text_embedder","retriever.query_embedding")
basic_rag_pipeline.connect("retriever","prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder","llm")

question = "What does Rhodes Statue look like?"
response = basic_rag_pipeline.run({"text_embedder": {"text": question}, 
                                   "prompt_builder": {"question": question}})
print("Response from Generator :", response["llm"]["replies"][0])

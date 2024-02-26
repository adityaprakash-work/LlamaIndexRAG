# ---DEPENDENCIES---------------------------------------------------------------
import os
import sys
import glob
import logging
import weaviate
from llama_index.core import StorageContext
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# --logging---------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ---weaviate client------------------------------------------------------------
client = weaviate.Client("http://localhost:8080")

# ---INDEXING-------------------------------------------------------------------
documents = SimpleDirectoryReader("./data/").load_data()

vector_store = WeaviateVectorStore(weaviate_client=client)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)


# ---QUERYING-------------------------------------------------------------------
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What is Aether?")
print(response)
# ---DEPENDENCIES---------------------------------------------------------------
import os
import sys
import glob
import torch
import logging
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
import weaviate
from llama_index.core import StorageContext
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# --logging---------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ---weaviate client------------------------------------------------------------
CLIENT = weaviate.Client("http://localhost:8080")

# ---constants------------------------------------------------------------------
DATA_DIR = "./data"
CACHE_FOLDER = "./model_cache"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ---embedding model------------------------------------------------------------
Settings.embed_model = HuggingFaceEmbedding(EMBEDDING_MODEL, cache_folder=CACHE_FOLDER)
Settings.llm = None

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
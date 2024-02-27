import sys
import glob
import logging
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
import weaviate
from llama_index.vector_stores.weaviate.base import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --logging---------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ---weaviate client------------------------------------------------------------
CLIENT = weaviate.Client("http://localhost:8083")

# ---constants------------------------------------------------------------------
DATA_DIR = "./data"
CACHE_FOLDER = "./model_cache"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ---embedding model------------------------------------------------------------
Settings.embed_model = HuggingFaceEmbedding(EMBEDDING_MODEL, cache_folder=CACHE_FOLDER)
Settings.llm = None

# ---INGEST---------------------------------------------------------------------
def ingest_data():
    with open("./exclude-files.txt", "r") as f:
        exclude = f.readlines()
    documents = SimpleDirectoryReader(DATA_DIR, exclude=exclude).load_data()
    exclude = glob.glob("./data/*")
    with open("./exclude-files.txt", "w") as f:
        f.truncate(0)
        for item in exclude:
            f.write("%s\n" % item)
    vector_store = WeaviateVectorStore(CLIENT)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    logging.info("Indexing complete.")

# ---MAIN-----------------------------------------------------------------------
if __name__ == "__main__":
    ingest_data()

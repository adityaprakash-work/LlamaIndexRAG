# ---DEPENDENCIES---------------------------------------------------------------
import os
import sys
import glob
import torch
import logging
import weaviate
from llama_index.core import StorageContext
from llama_index.vector_stores.weaviate.base import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import PromptTemplate


# --logging---------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ---weaviate client------------------------------------------------------------
client = weaviate.Client("http://localhost:8080")

# ---INDEXING-------------------------------------------------------------------
documents = SimpleDirectoryReader("./data/").load_data()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

Settings.llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    # No GPU
    device_map="cpu",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
)
# Settings.llm = None

vector_store = WeaviateVectorStore(weaviate_client=client)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)


# ---QUERYING-------------------------------------------------------------------
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What is Aether?")
print(response)
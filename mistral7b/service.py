# ---DEPENDENCIES---------------------------------------------------------------
import sys
import torch
import subprocess
import logging
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
import weaviate
from llama_index.vector_stores.weaviate.base import WeaviateVectorStore
from flask import Flask, request, jsonify

# --logging---------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ---weaviate client------------------------------------------------------------
CLIENT = weaviate.Client("http://localhost:8080")

# ---constants------------------------------------------------------------------
CACHE_FOLDER = "./model_cache"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'

# ---models---------------------------------------------------------------------
Settings.embed_model = HuggingFaceEmbedding(EMBEDDING_MODEL, cache_folder=CACHE_FOLDER)
Settings.llm = LlamaCPP(
    model_path=f'./model_cache/{LLM_MODEL}',
    temperature=0.1,
    max_new_tokens=256,
    context_window=4096,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# ---index----------------------------------------------------------------------
vector_store = WeaviateVectorStore(CLIENT)
index = VectorStoreIndex.from_vector_store(vector_store)
logging.info("Indexing complete.")

# ---chat engine----------------------------------------------------------------
chat_engine = index.as_chat_engine()

# ---APP------------------------------------------------------------------------
app = Flask(__name__)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    
    if user_input == ".exit":
        return jsonify({"response": "Exiting"})
    
    response = chat_engine.chat(user_input)
    
    return jsonify({"response": response})

@app.route('/ingest', methods=['POST'])
def ingest():
    subprocess.run(["python", "ingest.py"])
    return jsonify({"response": "Ingested data."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)

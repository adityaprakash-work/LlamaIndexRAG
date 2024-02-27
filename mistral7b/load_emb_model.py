# ---DEPENDENCIES---------------------------------------------------------------
import sys
import logging
# import urllib.request
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---logging--------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ---constants------------------------------------------------------------------
CACHE_FOLDER = "./model_cache"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
LLM_MODEL_URL = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf'

# ---MAIN-----------------------------------------------------------------------
if __name__ == "__main__":
    HuggingFaceEmbedding(EMBEDDING_MODEL, cache_folder=CACHE_FOLDER)
    # urllib.request.urlretrieve(LLM_MODEL_URL, f'{CACHE_FOLDER}/{LLM_MODEL}')
    logging.info("Models loaded.")



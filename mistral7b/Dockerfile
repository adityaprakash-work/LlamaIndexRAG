# Use the official Python 3.11 image as a base
FROM python:3.11-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the data and model cache folders from the host to the container
COPY data /app/data
COPY model_cache /app/model_cache

# Load model from Hugging Face URL into models_cache folder
RUN curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -o /app/model_cache/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Copy specific files
COPY exclude-files.txt ingest.py llama-index-req.txt service.py load_emb_model.py /app/

# Install dependencies in separate layers
RUN pip install torch
RUN pip install transformers
RUN pip install sentence-transformers
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN pip install -r llama-index-req.txt

# Run load_emb_model.py
RUN python load_emb_model.py

# Expose the required port
EXPOSE 5001

# Command to run the application
CMD ["python", "service.py"]

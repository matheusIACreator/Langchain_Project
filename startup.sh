#!/bin/bash

# Recria o vector store se não existir
if [ ! -d "data/vectorstore/renaissance" ]; then
    echo "🔧 Criando vector store..."
    python src/ingestion/pipeline.py
    python src/vectorstore.py
fi

# Inicia o servidor
python -m uvicorn api:app --host 0.0.0.0 --port 7860
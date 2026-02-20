"""
Configurações do Projeto RAG System - Galileu Galilei
Detecta automaticamente GPU/CPU e carrega variáveis de ambiente
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# ===== PATHS =====
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DATA_DIR / "chunks"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Cria diretórios se não existirem
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNKS_DIR, VECTORSTORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ===== DEVICE CONFIGURATION =====
def get_device() -> str:
    """
    Detecta automaticamente o melhor device disponível
    Prioridade: CUDA > MPS (Mac M1/M2) > CPU
    """
    # Verifica se há preferência manual no .env
    manual_device = os.getenv("DEVICE")
    if manual_device:
        print(f"⚙️  Device manual definido: {manual_device}")
        return manual_device
    
    # Detecta automaticamente
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Apple Silicon (MPS) detectado")
    else:
        device = "cpu"
        print("💻 Usando CPU (sem GPU disponível)")
    
    return device


DEVICE = get_device()


# ===== HUGGING FACE =====
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
    import warnings
    warnings.warn(
        "⚠️  HF_TOKEN não configurado — funcionalidades de LLM indisponíveis. "
        "Configure em .env para usar o modelo completo."
    )
    HF_TOKEN = None
# ===== MODEL CONFIGURATION =====
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Configuração de quantização
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "True").lower() == "true"
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "4"))  # 4 ou 8 bits

# Parâmetros do modelo LLM
if USE_QUANTIZATION and DEVICE == "cuda":
    print(f"🔧 Usando quantização {QUANTIZATION_BITS}-bit (QLoRA)")
    
    # Configurações para quantização
    if QUANTIZATION_BITS == 4:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal Float 4
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Nested quantization
        )
        
        MODEL_KWARGS = {
            "device_map": "auto",
            "quantization_config": bnb_config,
        }
    elif QUANTIZATION_BITS == 8:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        MODEL_KWARGS = {
            "device_map": "auto",
            "quantization_config": bnb_config,
        }
else:
    print(f"⚙️  Usando modelo sem quantização (FP16)")
    MODEL_KWARGS = {
        "device_map": "auto" if DEVICE == "cuda" else None,
        "torch_dtype": torch.float16,  # FP16 em CPU e GPU (~6GB RAM)
        "low_cpu_mem_usage": True,     # carrega shard a shard, não tudo de uma vez
    }

# Parâmetros de geração
GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}


# ===== CHROMADB CONFIGURATION =====
CHROMA_PERSIST_DIRECTORY = str(VECTORSTORE_DIR)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "galileu_collection")


# ===== DOCUMENT PROCESSING =====
# Configurações para chunking do PDF
CHUNK_SIZE = 1000  # Tamanho de cada chunk em caracteres
CHUNK_OVERLAP = 200  # Overlap entre chunks para manter contexto


# ===== RETRIEVAL CONFIGURATION =====
# Número de documentos a recuperar do vectorstore
TOP_K_DOCUMENTS = 4


# ===== MEMORY CONFIGURATION =====
# Número máximo de mensagens a manter na memória
MAX_MEMORY_MESSAGES = 10


# ===== PROMPT TEMPLATES =====
SYSTEM_PROMPT = """Você é um assistente especializado em Galileu Galilei, o pai da ciência moderna.

Você tem acesso a um documento detalhado sobre a vida, descobertas e contribuições de Galileu.

Diretrizes:
- Responda APENAS com base nas informações fornecidas no contexto
- Se a pergunta não puder ser respondida com o contexto disponível, diga claramente
- Seja preciso e cite datas, nomes e eventos específicos quando relevante
- Mantenha um tom educativo e acessível
- Se perguntarem sobre algo além de Galileu, redirecione educadamente ao tópico

Contexto:
{context}

Histórico da conversa:
{chat_history}

Pergunta: {question}

Resposta:"""


# ===== DEBUG MODE =====
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

if DEBUG:
    print("\n" + "="*50)
    print("🔧 CONFIGURAÇÕES DO PROJETO")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Quantization: {USE_QUANTIZATION}")
    if USE_QUANTIZATION and DEVICE == "cuda":
        print(f"Quantization Bits: {QUANTIZATION_BITS}-bit")
    print(f"Chroma Directory: {CHROMA_PERSIST_DIRECTORY}")
    print(f"Collection Name: {COLLECTION_NAME}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Top K Documents: {TOP_K_DOCUMENTS}")
    print("="*50 + "\n")
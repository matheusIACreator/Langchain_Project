# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Evita prompts interativos durante apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Dependências do sistema ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Usuário não-root (exigido pelo HF Spaces) ────────────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# ── Diretório de trabalho ────────────────────────────────────────────────────
WORKDIR /home/user/app

# ── Dependências Python ──────────────────────────────────────────────────────
COPY --chown=user requirements.txt .

# Instala PyTorch com CUDA 11.8
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Instala o resto das dependências
RUN pip install --no-cache-dir -r requirements.txt

# ── Código da aplicação ──────────────────────────────────────────────────────
COPY --chown=user . .

# ── Variáveis de ambiente padrão ─────────────────────────────────────────────
# HF_TOKEN deve ser configurado nas Secrets do Space
ENV MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
ENV EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENV USE_QUANTIZATION="True"
ENV QUANTIZATION_BITS="4"
ENV DEBUG="False"

# Cache do Hugging Face
ENV HF_HOME=/home/user/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface

# ── Porta (HF Spaces exige 7860) ─────────────────────────────────────────────
EXPOSE 7860

# ── Comando de inicialização ─────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
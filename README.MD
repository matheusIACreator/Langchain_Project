# 🔭 Science Chat — Historical Figures RAG System

> **Sistema avançado de Retrieval-Augmented Generation (RAG) para explorar a história da ciência através de conversas inteligentes com múltiplas figuras históricas.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)]()
[![Status](https://img.shields.io/badge/status-v2.1--deployed-brightgreen.svg)]()

**🌐 Demo ao vivo:** [langchain-project-nine.vercel.app](https://langchain-project-nine.vercel.app)  
**⚙️ API:** [shinigami4242557-science-chat-api.hf.space](https://shinigami4242557-science-chat-api.hf.space)

---

## 📖 Sobre o Projeto

Este projeto começou como um chatbot focado em **Galileu Galilei** e evoluiu para um **sistema escalável** capaz de responder perguntas sobre múltiplas figuras históricas através de diferentes períodos da ciência — com frontend moderno, backend em produção e pipeline RAG híbrido completo.

### 🎯 O Que o Sistema Faz

- 💬 **Conversas Naturais**: Faça perguntas em português ou inglês
- 🧠 **Roteamento Inteligente**: Detecta automaticamente qual figura/período é relevante
- 🔍 **Busca Híbrida**: Combina busca semântica (conceitos) e keyword (nomes/datas)
- 📚 **Multi-Período**: Renascimento, Iluminismo e Era Moderna
- 🤝 **Comparações**: Compare contribuições de diferentes cientistas
- 💾 **Memória Conversacional**: Mantém contexto da conversa
- ⚡ **Multilingual**: Embeddings que suportam PT/EN/50+ idiomas

---

## 🚀 Demonstração

```
Você: "Quando Galileu nasceu?"
Bot: "Galileu Galilei nasceu em 15 de fevereiro de 1564, em Pisa, Itália..."

Você: "E Newton?"
Bot: "Isaac Newton nasceu em 25 de dezembro de 1642, em Woolsthorpe, Inglaterra..."

Você: "Compare as teorias de gravidade de Newton e Einstein"
Bot: "Newton desenvolveu a lei da gravitação universal... Einstein revolucionou esse conceito com a teoria da relatividade geral, propondo que a gravidade é a curvatura do espaço-tempo..."
```

---

## 📊 Versões do Projeto

### ✅ v2.1 — Deployed (Estado Atual)
**Status**: ✅ Em produção

- Sistema multi-figura com 3 cientistas históricos
- Embedding multilingual (`paraphrase-multilingual-MiniLM-L12-v2`)
- LLM: Llama 3.2 3B Instruct
- Frontend Next.js deployado na Vercel
- Backend FastAPI deployado no HuggingFace Spaces
- Hybrid retrieval (BM25 + dense + RRF)
- 5 camadas de testes com MRR = 1.0

### 📌 v1.0 — Single Figure (Histórico)
**Status**: ✅ Estável

- Chatbot focado exclusivamente em Galileu Galilei
- Vector store com collection única
- Interface Gradio

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────┐
│     Frontend Next.js (Vercel)                │
│     langchain-project-nine.vercel.app        │
└──────────────────┬──────────────────────────┘
                   │ HTTP
         ┌─────────▼──────────┐
         │  FastAPI Backend   │ ← HuggingFace Spaces
         │  /chat /health     │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │   Topic Router     │ ← Detecta figura/período/expert
         └─────────┬──────────┘
                   │
    ┌──────────────┴─────────────────┐
    │                                │
┌───▼──────────┐         ┌──────────▼────┐
│ Dense Search │         │ Sparse Search │
│ (Embeddings) │         │    (BM25)     │
└───┬──────────┘         └──────────┬────┘
    │                               │
    └───────────┬───────────────────┘
                │ Reciprocal Rank Fusion
    ┌───────────▼────────────────────┐
    │  Multi-Collection ChromaDB     │
    │  renaissance/galileo_galilei   │
    │  enlightenment/isaac_newton    │
    │  modern_era/albert_einstein    │
    └───────────┬────────────────────┘
                │
        ┌───────▼────────┐
        │ Llama 3.2 3B   │ ← CPU (HF Spaces)
        │ + Memory k=5   │
        └───────┬────────┘
                │
    ┌───────────▼───────────────┐
    │   Contextualized Answer   │
    └───────────────────────────┘
```

### 🔑 Componentes Principais

#### 1. **Topic Router** (`src/retrieval/topic_router.py`)
Analisa queries e determina qual expert deve responder (Physics, Biography, Astronomy, etc), quais figuras estão sendo mencionadas e quais períodos históricos são relevantes.

#### 2. **Multi-Collection Vector Store** (`src/vectorstore.py`)
Gerencia múltiplas collections no ChromaDB com estrutura `period/figure`. Suporta lazy loading e busca em uma ou múltiplas collections simultaneamente.

#### 3. **Hybrid Retriever** (`src/retrieval/hybrid_retriever.py`)
Combina Dense Search (embeddings semânticos), Sparse Search (BM25) e Reciprocal Rank Fusion para ranking final.

#### 4. **Ingestion Pipeline** (`src/ingestion/pipeline.py`)
Processa biografias automaticamente, divide em chunks de 1000 chars (overlap 200), adiciona metadata rica e organiza por período histórico.

#### 5. **Conversation Memory** (`src/memory/conversation_memory.py`)
Window buffer das últimas 5 interações para manter contexto de follow-ups.

---

## 📚 Figuras Históricas Disponíveis

| Figura | Período | Anos | Chunks | Fonte |
|--------|---------|------|--------|-------|
| 🔭 Galileu Galilei | Renascimento | 1564–1642 | 96 | Wikipedia EN |
| 🍎 Isaac Newton | Iluminismo | 1643–1727 | 127 | Wikipedia EN |
| ⚛️ Albert Einstein | Era Moderna | 1879–1955 | 126 | Wikipedia EN |

**Total**: 3 figuras | 349 chunks | ~243KB de conhecimento histórico

### 📅 Planejadas para Futuras Versões

**Renascimento**: Leonardo da Vinci, Copérnico  
**Iluminismo**: Benjamin Franklin, René Descartes  
**Era Moderna**: Marie Curie, Charles Darwin, Nikola Tesla, Richard Feynman

---

## 🛠️ Tecnologias Utilizadas

**Backend**: FastAPI, LangChain 0.3+, ChromaDB, LangChain-Chroma  
**LLM**: Meta Llama 3.2 3B Instruct  
**Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (50+ idiomas)  
**Retrieval**: BM25 (rank-bm25) + Dense embeddings + RRF  
**Frontend**: Next.js 14, TypeScript, shadcn/ui  
**Deploy**: HuggingFace Spaces (backend) + Vercel (frontend)

---

## 📦 Instalação Local

### Pré-requisitos

- Python 3.11+
- GPU NVIDIA com CUDA 11.8+ (recomendado) ou CPU
- Token do HuggingFace com acesso ao Llama 3.2
- Node.js 18+ (para o frontend)

### Backend

```bash
# 1. Clone o repositório
git clone https://github.com/matheusIACreator/Langchain_Project.git
cd Langchain_Project

# 2. Crie ambiente virtual
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac

# 3. Instale PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Instale dependências
pip install -r requirements.txt

# 5. Configure o .env
cp .env.example .env
# Edite com seu HF_TOKEN e configurações

# 6. Crie o vector store
python src/ingestion/pipeline.py
python src/vectorstore.py

# 7. Inicie o servidor
uvicorn api:app --reload --port 8000
```

### Frontend

```bash
cd src/frontend/science-chat
npm install
npm run dev
# Acesse http://localhost:3000
```

### Variáveis de Ambiente

```dotenv
HF_TOKEN=seu_token_aqui
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
USE_QUANTIZATION=True
QUANTIZATION_BITS=4
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🌍 Deploy

### Backend — HuggingFace Spaces

O backend roda em container Docker com CPU básica. No primeiro startup, o vector store é criado automaticamente via `startup.sh`.

```bash
git remote add space https://huggingface.co/spaces/Shinigami4242557/science-chat-api
git push space main
```

Variável necessária no Space (Settings → Secrets):
```
HF_TOKEN=seu_token_aqui
```

### Frontend — Vercel

Deploy automático via GitHub. Variável necessária no Vercel:
```
NEXT_PUBLIC_API_URL=https://shinigami4242557-science-chat-api.hf.space
```

---

## 🔌 API Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/figures` | Lista figuras disponíveis |
| POST | `/chat` | Envia mensagem e recebe resposta |
| DELETE | `/conversation/{session_id}` | Limpa histórico da sessão |
| GET | `/stats` | Estatísticas do sistema |

### Exemplo

```bash
curl -X POST https://shinigami4242557-science-chat-api.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quando Galileu nasceu?"}'
```

---

## 📁 Estrutura do Projeto

```
Langchain_Project/
├── api.py                          # FastAPI — endpoints principais
├── startup.sh                      # Script de inicialização do container
├── Dockerfile                      # Deploy HF Spaces
├── requirements.txt
├── config/
│   └── settings.py                 # Device, modelo, prompts, geração
├── data/
│   └── raw/                        # Biografias em texto (Wikipedia)
│       ├── renaissance/galileo_galilei.txt
│       ├── enlightenment/isaac_newton.txt
│       └── modern_era/albert_einstein.txt
├── src/
│   ├── ingestion/pipeline.py       # Chunking e processamento
│   ├── vectorstore.py              # MultiCollectionVectorStore
│   ├── retrieval/
│   │   ├── topic_router.py         # Roteamento inteligente
│   │   └── hybrid_retriever.py     # BM25 + dense + RRF
│   ├── chains/
│   │   └── rag_chain_multi.py      # Chain Multi-Figura v2.1
│   ├── memory/
│   │   └── conversation_memory.py
│   └── frontend/
│       └── science-chat/           # Next.js
├── tests/
│   ├── test_e2e.py                 # 4 camadas de testes
│   └── test_retrieval_metrics.py   # MRR e NDCG
└── scripts/
    └── download_wikipedia_api.py   # Ingestão via Wikipedia API
```

---

## 🧪 Testes e Métricas

O sistema foi avaliado com 5 camadas de testes antes do deploy:

| Camada | Escopo | Resultado |
|--------|--------|-----------|
| 1 — Topic Router | Roteamento sem GPU | ✅ |
| 2 — Memória | Conversação sem GPU | ✅ |
| 3 — Hybrid Retriever | Retrieval sem LLM | ✅ |
| 4 — Chain E2E | Pipeline completo | ✅ |
| 5 — MRR / NDCG | Qualidade do retrieval | MRR = 1.0 |

```bash
# Testes sem LLM (rápido)
pytest tests/test_e2e.py -v -m "not llm"

# Métricas de retrieval
pytest tests/test_retrieval_metrics.py -v

# Tudo incluindo LLM
pytest tests/test_e2e.py -v
```

---

## ⚙️ Configuração Avançada

### GPU / CPU

```python
# Detecção automática: CUDA > MPS > CPU
# Para forçar device no .env:
DEVICE=cuda  # ou 'cpu', 'mps'
```

### Quantização (QLoRA)

| Configuração | VRAM | Qualidade | Velocidade |
|--------------|------|-----------|------------|
| FP16 (sem quantização) | ~6GB | 100% | 100% |
| 8-bit (INT8) | ~4GB | ~98% | ~85% |
| **4-bit (NF4/QLoRA)** | **~2GB** | **~95%** | **~70%** |

### Parâmetros de Geração

```python
# Em config/settings.py:
GENERATION_KWARGS = {
    "max_new_tokens": 1024,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.2,
}
```

---

## 🐛 Troubleshooting

**CUDA não detectado:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Se False: reinstalar com --index-url https://download.pytorch.org/whl/cu118
```

**Vector store não encontrado:**
```bash
python src/ingestion/pipeline.py
python src/vectorstore.py
```

**Respostas repetitivas:**
```python
# Aumentar repetition_penalty em config/settings.py
"repetition_penalty": 1.3
```

**Respostas em inglês para queries em português:**
```python
# Verificar EMBEDDING_MODEL em settings.py
# Deve ser: paraphrase-multilingual-MiniLM-L12-v2
```

---

## 🗺️ Roadmap

- [x] Sistema RAG básico (Galileu)
- [x] Arquitetura multi-figura
- [x] Hybrid retrieval (BM25 + dense)
- [x] Embedding multilingual
- [x] Frontend Next.js
- [x] Deploy (HF Spaces + Vercel)
- [x] Testes E2E com MRR/NDCG
- [ ] Streaming de respostas
- [ ] Mais figuras históricas (Darwin, Curie, Tesla)
- [ ] Histórico de conversas persistente
- [ ] Backend em GPU para respostas mais rápidas

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/NovaFigura`)
3. Commit (`git commit -m 'Add: Marie Curie biography'`)
4. Push (`git push origin feature/NovaFigura`)
5. Abra um Pull Request

---

## 📚 Referências

- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

## 📚 Documentação Adicional

- **[SCALING_PLAN.md](docs/SCALING_PLAN.md)**: Arquitetura completa e estratégia de scaling
- **[QLORA.MD](QLORA.MD)**: Guia de quantização e otimização de memória

---

## 📝 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.

---

## 👨‍💻 Autor

**Matheus Masago**

- 🌐 GitHub: [@matheusIACreator](https://github.com/matheusIACreator)
- 💼 LinkedIn: [Matheus Masago](https://www.linkedin.com/in/matheus-masago/)

---

## 🙏 Agradecimentos

- **[LangChain](https://www.langchain.com/)** — Framework RAG
- **[Hugging Face](https://huggingface.co/)** — Modelos e Embeddings
- **[ChromaDB](https://www.trychroma.com/)** — Vector Database
- **[Meta AI](https://ai.meta.com/)** — Llama Models
- **[Anthropic](https://www.anthropic.com/)** — Claude AI (development assistance)
- **[Wikipedia](https://www.wikipedia.org/)** — Fonte de dados históricos

---

<div align="center">

**⚡ Construído com Python, LangChain, Next.js e muita ☕**

[⬆ Voltar ao topo](#-science-chat--historical-figures-rag-system)

</div>
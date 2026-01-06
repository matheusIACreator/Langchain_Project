# 🔭 Historical Figures RAG System

> **Sistema avançado de Retrieval-Augmented Generation (RAG) para explorar a história da ciência através de conversas inteligentes com múltiplas figuras históricas.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/license-Educational-orange.svg)]()
[![Status](https://img.shields.io/badge/status-v2.0--dev-yellow.svg)]()

---

## 📖 Sobre o Projeto

Este projeto começou como um chatbot focado em **Galileu Galilei** e evoluiu para um **sistema escalável** capaz de responder perguntas sobre múltiplas figuras históricas através de diferentes períodos da ciência.

### 🎯 O Que o Sistema Faz

- 💬 **Conversas Naturais**: Faça perguntas sobre cientistas históricos em linguagem natural
- 🧠 **Roteamento Inteligente**: Sistema detecta automaticamente qual figura/período é relevante
- 🔍 **Busca Híbrida**: Combina busca semântica (conceitos) e keyword (nomes/datas)
- 📚 **Multi-Período**: Suporta figuras do Renascimento, Iluminismo e Era Moderna
- 🤝 **Comparações**: Compare contribuições de diferentes cientistas
- 💾 **Memória Conversacional**: Mantém contexto da conversa
- ⚡ **Eficiente**: Roda em GPU de 4GB usando quantização 4-bit

---

## 🚀 Demonstração

```
Você: "Quando Galileu nasceu?"
Bot: "Galileu Galilei nasceu em 15 de fevereiro de 1564, em Pisa, Itália..."

Você: "E Newton?"
Bot: "Isaac Newton nasceu em 25 de dezembro de 1642 (4 de janeiro de 1643 no calendário gregoriano), em Woolsthorpe, Inglaterra..."

Você: "Compare as teorias de gravidade de Newton e Einstein"
Bot: "Newton desenvolveu a lei da gravitação universal, descrevendo a gravidade como uma força de atração entre massas... Einstein, por outro lado, revolucionou esse conceito com a teoria da relatividade geral, propondo que a gravidade não é uma força, mas sim a curvatura do espaço-tempo causada pela presença de massa..."
```

---

## 📊 Versões do Projeto

### 📌 v1.0 - Single Figure (Branch: `main`)
**Status**: ✅ Estável e Funcional

- Chatbot focado exclusivamente em **Galileu Galilei**
- Vector store com collection única
- Interface Gradio básica
- Memória conversacional

### 🚧 v2.0 - Multi-Figure System (Branch: `feature/multi-figure-scaling`)
**Status**: 🏗️ Em Desenvolvimento (80% completo)

- Sistema escalável para **múltiplas figuras históricas**
- Arquitetura multi-collection (ChromaDB)
- Topic routing inteligente
- Hybrid retrieval (dense + sparse)
- Suporte a 3 períodos históricos

**📈 Progresso v2.0:**
```
[████████████████████░░] 80%

✅ Arquitetura documentada
✅ Multi-collection vector store
✅ Topic router
✅ Hybrid retrieval
✅ Pipeline de ingestão
✅ 3 figuras implementadas
🚧 Interface atualizada
📅 Mixture of Experts
📅 Métricas de avaliação
```

---

## 🏗️ Arquitetura do Sistema (v2.0)

```
┌─────────────────────────────────────────────┐
│           User Query                         │
│   "Compare Newton e Einstein"                │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   Topic Router     │ ← Detecta figuras/período/expert
         │  • Physics Expert  │
         │  • Biography       │
         │  • Astronomy       │
         └─────────┬──────────┘
                   │
    ┌──────────────┴─────────────────┐
    │                                │
┌───▼──────────┐         ┌──────────▼────┐
│ Dense Search │         │ Sparse Search │
│ (Semantic)   │         │    (BM25)     │
└───┬──────────┘         └──────────┬────┘
    │                               │
    └───────────┬───────────────────┘
                │ Hybrid Retrieval
    ┌───────────▼────────────────────┐
    │  Multi-Collection VectorStore  │
    │  ┌─────────────────────────┐   │
    │  │ renaissance/            │   │
    │  │   • galileo_galilei     │   │
    │  ├─────────────────────────┤   │
    │  │ enlightenment/          │   │
    │  │   • isaac_newton        │   │
    │  ├─────────────────────────┤   │
    │  │ modern_era/             │   │
    │  │   • albert_einstein     │   │
    │  └─────────────────────────┘   │
    └───────────┬────────────────────┘
                │
        ┌───────▼────────┐
        │  LLM + Memory  │ ← Llama-3.1-8B (4-bit)
        │  Generate      │
        └───────┬────────┘
                │
    ┌───────────▼───────────────┐
    │   Contextualized Answer   │
    └───────────────────────────┘
```

### 🔑 Componentes Principais

#### 1. **Topic Router** (`src/retrieval/topic_router.py`)
Analisa queries e determina:
- Qual expert deve responder (Physics, Biography, Astronomy, etc)
- Quais figuras estão sendo mencionadas
- Quais períodos históricos são relevantes

#### 2. **Multi-Collection Vector Store** (`src/vectorstore.py`)
- Gerencia múltiplas collections no ChromaDB
- Estrutura: `period/figure` (ex: `renaissance/galileo_galilei`)
- Lazy loading para eficiência de memória
- Busca em uma ou múltiplas collections

#### 3. **Hybrid Retriever** (`src/retrieval/hybrid_retriever.py`)
- **Dense Search**: Busca semântica usando embeddings
- **Sparse Search**: Busca por keywords usando BM25
- **Reciprocal Rank Fusion**: Combina ambos os rankings

#### 4. **Ingestion Pipeline** (`src/ingestion/pipeline.py`)
- Processa biografias automaticamente
- Divide em chunks inteligentes
- Adiciona metadata rica
- Organiza por período histórico

#### 5. **Conversation Memory** (`src/memory/conversation_memory.py`)
- Buffer window de últimas 5 interações
- Mantém contexto da conversa
- Permite perguntas de follow-up

---

## 📚 Figuras Históricas Disponíveis

### ✅ Renascimento (1400-1600)
| Figura | Status | Chunks | Fonte |
|--------|--------|--------|-------|
| **Galileo Galilei** | ✅ Implementado | 96 | Wikipedia EN |

### ✅ Iluminismo (1650-1800)
| Figura | Status | Chunks | Fonte |
|--------|--------|--------|-------|
| **Isaac Newton** | ✅ Implementado | 127 | Wikipedia EN |

### ✅ Era Moderna (1800-1950)
| Figura | Status | Chunks | Fonte |
|--------|--------|--------|-------|
| **Albert Einstein** | ✅ Implementado | 126 | Wikipedia EN |

### 📅 Planejadas para Futuras Versões

**Renascimento:**
- Leonardo da Vinci
- Michelangelo
- Nicolau Copérnico

**Iluminismo:**
- Voltaire
- Benjamin Franklin
- René Descartes

**Era Moderna:**
- Marie Curie
- Charles Darwin
- Nikola Tesla
- Richard Feynman

**Total Atual**: 3 figuras | 349 chunks | ~242KB de conhecimento histórico

---

## 🛠️ Tecnologias Utilizadas

### Core Stack
- **LLM**: Meta Llama-3.1-8B-Instruct (4-bit quantização)
- **Vector Database**: ChromaDB (multi-collection)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Framework**: LangChain 0.3.0+
- **Interface**: Gradio 4.0+

### Retrieval
- **Dense Search**: Embeddings semânticos (384 dimensões)
- **Sparse Search**: BM25 (keyword matching)
- **Fusion**: Reciprocal Rank Fusion

### Otimizações
- **Quantização**: QLoRA 4-bit (bitsandbytes)
- **Device**: Auto-detect (CUDA > MPS > CPU)
- **Memory**: Lazy loading de collections

---

## 📦 Instalação

### Pré-requisitos

- Python 3.9+ (recomendado: 3.11 ou 3.12)
- GPU NVIDIA com CUDA 11.8+ (opcional, mas recomendado)
- 4GB+ VRAM (com quantização) ou 16GB+ (sem quantização)
- 10GB+ espaço em disco

### 1. Clone o Repositório

```bash
git clone https://github.com/matheusIACreator/Langchain_Project.git
cd Langchain_Project
```

### 2. Escolha a Versão

**Para v1.0 (Galileu apenas - estável):**
```bash
git checkout main
```

**Para v2.0 (Multi-figura - desenvolvimento):**
```bash
git checkout feature/multi-figure-scaling
```

### 3. Crie Ambiente Virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 4. Instale Dependências

```bash
# Dependências principais
pip install -r requirements.txt

# Para GPU NVIDIA (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Configure Variáveis de Ambiente

```bash
# Copie o template
cp .env.example .env

# Edite e adicione seu token Hugging Face
# HF_TOKEN=seu_token_aqui
```

**Obtenha seu token**: https://huggingface.co/settings/tokens

### 6. Prepare os Dados

**Opção A: Usar dados existentes (v2.0)**
```bash
# Já incluído no branch feature/multi-figure-scaling
# Biografias de Galileu, Newton e Einstein
```

**Opção B: Baixar novas figuras (v2.0)**
```bash
python scripts/download_wikipedia_api.py --figures "Charles Darwin,Marie Curie" --language en --output-dir data/raw
```

---

## 🎯 Como Usar

### v1.0 - Chatbot do Galileu

```bash
# 1. Processar documento
python src/document_loader.py

# 2. Criar vector store
python src/vectorstore.py --mode single

# 3. Executar chatbot
python main.py
```

### v2.0 - Sistema Multi-Figura

```bash
# 1. Processar todas as biografias
python src/ingestion/pipeline.py

# 2. Criar multi-collection vector store
python src/vectorstore.py --mode multi

# 3. Testar RAG chain completa
python src/chains/rag_chain_multi.py

# 4. Executar interface (em desenvolvimento)
python main.py
```

### Testes Individuais de Componentes

```bash
# Testar Topic Router
python src/retrieval/topic_router.py

# Testar Hybrid Retriever
python src/retrieval/hybrid_retriever.py

# Baixar novas biografias
python scripts/download_wikipedia_api.py --period enlightenment --language en
```

---

## 📁 Estrutura do Projeto

```
Langchain_Project/
├── .env                        # Variáveis de ambiente (não versionado)
├── .env.example               # Template de configuração
├── .gitignore                 
├── README.md                  # Este arquivo
├── requirements.txt           # Dependências Python
├── main.py                    # Interface principal (Gradio)
│
├── config/
│   ├── settings.py            # Configurações gerais do sistema
│   └── experts_config.py      # Configuração de experts (v2.0)
│
├── data/
│   ├── raw/                   # Biografias organizadas por período
│   │   ├── renaissance/
│   │   │   └── galileo_galilei.txt (96 chunks)
│   │   ├── enlightenment/
│   │   │   └── isaac_newton.txt (127 chunks)
│   │   └── modern_era/
│   │       └── albert_einstein.txt (126 chunks)
│   │
│   ├── processed/             # Dados processados
│   │   └── chunks/
│   │
│   └── vectorstore/           # ChromaDB multi-collection
│       ├── renaissance/
│       │   └── galileo_galilei/
│       ├── enlightenment/
│       │   └── isaac_newton/
│       └── modern_era/
│           └── albert_einstein/
│
├── src/
│   ├── document_loader.py    # Processador de PDFs (v1.0)
│   ├── vectorstore.py         # Vector store manager (v1.0 + v2.0)
│   │
│   ├── retrieval/             # Sistema de retrieval (v2.0)
│   │   ├── topic_router.py           # Roteamento inteligente
│   │   └── hybrid_retriever.py       # Dense + Sparse search
│   │
│   ├── ingestion/             # Pipeline de dados (v2.0)
│   │   └── pipeline.py               # Processamento automatizado
│   │
│   ├── chains/                # RAG Chains
│   │   ├── rag_chain.py              # Single-figure chain (v1.0)
│   │   └── rag_chain_multi.py        # Multi-figure chain (v2.0)
│   │
│   ├── prompts/
│   │   └── rag_prompts.py            # Templates de prompts
│   │
│   ├── memory/
│   │   └── conversation_memory.py    # Sistema de memória
│   │
│   └── utils/
│       └── helpers.py
│
├── scripts/                   # Scripts utilitários (v2.0)
│   └── download_wikipedia_api.py     # Download de biografias
│
├── docs/                      # Documentação detalhada (v2.0)
│   ├── SCALING_PLAN.md               # Arquitetura completa
│   └── NEXT_STEPS.md                 # Guia de implementação
│
├── notebooks/                 # Jupyter notebooks exploratórios
│   ├── 01_data_exploration.ipynb
│   ├── 02_vectorstore_setup.ipynb
│   └── 03_rag_testing.ipynb
│
└── tests/                     # Testes unitários
    └── test_*.py
```

---

## ⚙️ Configuração Avançada

### GPU / CPU

O sistema detecta automaticamente o melhor device:

```python
# Prioridade: CUDA > MPS (Mac M1/M2) > CPU

# Para forçar um device específico:
# No .env:
DEVICE=cuda  # ou 'cpu', 'mps'
```

### Quantização (QLoRA)

Para GPUs com 4GB VRAM (como RTX 3050):

```python
# No .env:
USE_QUANTIZATION=True
QUANTIZATION_BITS=4  # 4-bit ou 8-bit
```

**Comparação de uso de memória:**

| Configuração | VRAM | Qualidade | Velocidade |
|--------------|------|-----------|------------|
| FP16 (sem quantização) | ~16GB | 100% | 100% |
| 8-bit (INT8) | ~8GB | ~98% | ~85% |
| **4-bit (NF4/QLoRA)** | **~4GB** | **~95%** | **~70%** |

### Chunking

Ajuste o tamanho dos chunks:

```python
# Em config/settings.py:
CHUNK_SIZE = 1000        # Caracteres por chunk
CHUNK_OVERLAP = 200      # Overlap para manter contexto
```

### Retrieval

Número de documentos recuperados:

```python
# Em config/settings.py:
TOP_K_DOCUMENTS = 4      # Documentos por busca
```

### Memory

Tamanho da janela de memória:

```python
# Em src/chains/rag_chain_multi.py:
self.memory = GalileuConversationMemory(
    memory_type="window",
    k=5  # Últimas 5 interações
)
```

---

## 🧪 Exemplos de Uso

### Single-Figure Query (v1.0)

```python
from src.chains.rag_chain import GalileuRAGChain

chain = GalileuRAGChain()

response = chain.chat("Quais foram as principais descobertas de Galileu?")
print(response)
# Output: "Galileu Galilei fez várias descobertas revolucionárias..."
```

### Multi-Figure Query (v2.0)

```python
from src.chains.rag_chain_multi import MultiFigureRAGChain

chain = MultiFigureRAGChain()

# Query sobre figura específica
response = chain.chat("Quando Newton nasceu?")

# Query comparativa
response = chain.chat("Compare as teorias de gravidade de Newton e Einstein")

# Query com follow-up (usa memória)
response = chain.chat("Quem foi Galileu?")
response = chain.chat("Quais foram suas principais invenções?")  # Sabe que é sobre Galileu
```

### Download de Novas Figuras

```bash
# Download individual
python scripts/download_wikipedia_api.py \
    --figures "Marie Curie" \
    --language en \
    --output-dir data/raw/modern_era

# Download por período
python scripts/download_wikipedia_api.py \
    --period enlightenment \
    --language en

# Download de múltiplas figuras
python scripts/download_wikipedia_api.py \
    --figures "Charles Darwin,Nikola Tesla,Marie Curie" \
    --language en
```

---

## 📊 Estatísticas e Métricas

### Dados Processados (v2.0)

```
3 períodos históricos
3 figuras científicas
349 chunks totais
~242KB de texto original
~126,000 embeddings (349 chunks × 384 dimensões)
```

### Performance

```
Busca semântica: ~50ms
Busca BM25: ~20ms
Hybrid fusion: ~10ms
LLM generation (4-bit): ~2-3s
Total query end-to-end: ~3-4s
```

### Modelo

```
LLM: Llama-3.1-8B-Instruct
Quantização: 4-bit NF4
VRAM Usage: ~4GB
Embeddings: all-MiniLM-L6-v2 (384 dims)
Vector DB: ChromaDB (SQLite backend)
```

---

## 🚦 Roadmap

### ✅ Fase 1: Fundação (Concluída)
- [x] Sistema RAG básico com Galileu
- [x] Arquitetura de scaling documentada
- [x] Topic routing implementado
- [x] Hybrid retrieval implementado
- [x] Multi-collection vector store
- [x] Pipeline de ingestão
- [x] 3 figuras históricas adicionadas

### 🚧 Fase 2: Integração (Em Progresso - 80%)
- [x] RAG Chain multi-figura
- [x] Roteamento integrado
- [x] Memória conversacional
- [ ] Interface Gradio atualizada
- [ ] Testes end-to-end completos

### 📅 Fase 3: Mixture of Experts (Planejado)
- [ ] Implementar sistema MoE
- [ ] Fine-tune experts especializados
  - Physics Expert
  - Biography Expert
  - Astronomy Expert
- [ ] Lazy loading de modelos
- [ ] Benchmarks de performance

### 📅 Fase 4: Qualidade e Produção (Planejado)
- [ ] Métricas de precisão histórica
- [ ] Validação de citações
- [ ] Sistema de feedback do usuário
- [ ] Interface web aprimorada
- [ ] Deploy (Docker + API REST)
- [ ] Documentação completa de API

### 📅 Fase 5: Expansão (Futuro)
- [ ] 10-15 figuras históricas
- [ ] Suporte multilingual (PT, ES, FR)
- [ ] Busca temporal (timeline)
- [ ] Visualizações interativas
- [ ] Mobile app

---

## 🧠 Conceitos Técnicos Explicados

### O que é RAG?

**Retrieval-Augmented Generation** combina:
1. **Retrieval**: Busca de informações relevantes em base de conhecimento
2. **Generation**: LLM gera resposta usando as informações encontradas

**Vantagens sobre LLM puro:**
- ✅ Informações atualizadas (não limitado ao training cutoff)
- ✅ Cita fontes específicas
- ✅ Menos alucinações
- ✅ Controle sobre conhecimento usado

### Dense vs Sparse Retrieval

**Dense (Semantic Search):**
- Usa embeddings vetoriais (384 dimensões)
- Entende significado e contexto
- Bom para: queries conceituais, sinônimos

**Sparse (BM25):**
- Usa term frequency × inverse document frequency
- Busca palavras exatas
- Bom para: nomes próprios, datas, termos específicos

**Hybrid = Melhor dos dois mundos!**

### Por que Multi-Collection?

**Single Collection** (todos juntos):
```
❌ Mistura contextos de diferentes figuras
❌ Busca menos precisa
❌ Difícil manter/atualizar
```

**Multi-Collection** (separados):
```
✅ Contextos isolados
✅ Busca mais precisa
✅ Fácil adicionar/remover figuras
✅ Permite comparações controladas
```

### Conversation Memory

Mantém contexto para perguntas de follow-up:

```
User: "Quando Galileu nasceu?"
Bot: "Em 1564"
User: "E onde ele morreu?"  ← Sabe que "ele" = Galileu
Bot: "Em Arcetri, perto de Florença"
```

**Window Memory (k=5)**: Mantém últimas 5 interações para não estourar limite de tokens.

---

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"

**Solução 1**: Usar quantização 4-bit
```bash
# No .env:
USE_QUANTIZATION=True
QUANTIZATION_BITS=4
```

**Solução 2**: Reduzir batch size
```python
# Em config/settings.py:
GENERATION_KWARGS = {
    "max_new_tokens": 256,  # Reduzir de 512
}
```

### Erro: "Vector store not found"

```bash
# Recriar vector store:
python src/vectorstore.py --mode multi
```

### Erro: "HF_TOKEN not found"

```bash
# 1. Verificar .env existe
cat .env

# 2. Adicionar token
echo "HF_TOKEN=seu_token_aqui" >> .env

# 3. Obter token em: https://huggingface.co/settings/tokens
```

### Performance Lenta

**Causas comuns:**
- CPU mode (sem GPU)
- Modelo sem quantização
- Muitos documentos sendo recuperados

**Soluções:**
```bash
# 1. Verificar device
python -c "import torch; print(torch.cuda.is_available())"

# 2. Ativar quantização (ver seção Configuração)

# 3. Reduzir TOP_K_DOCUMENTS em config/settings.py
```

### Respostas Genéricas / Alucinações

**Causas:**
- Poucos documentos relevantes
- Chunks muito pequenos/grandes
- LLM não seguindo contexto

**Soluções:**
```python
# 1. Aumentar TOP_K_DOCUMENTS
TOP_K_DOCUMENTS = 6  # Era 4

# 2. Ajustar chunk size
CHUNK_SIZE = 1200  # Era 1000
CHUNK_OVERLAP = 250  # Era 200

# 3. Melhorar prompt (src/prompts/rag_prompts.py)
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! 

### Como Contribuir

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFigura`)
3. Commit suas mudanças (`git commit -m 'Add: Marie Curie biography'`)
4. Push para a branch (`git push origin feature/NovaFigura`)
5. Abra um Pull Request

### Tipos de Contribuições

- 📚 Adicionar novas figuras históricas
- 🐛 Reportar e corrigir bugs
- 📝 Melhorar documentação
- ✨ Propor novas features
- 🧪 Adicionar testes
- 🎨 Melhorar interface

### Guidelines

- Código bem documentado (docstrings)
- Seguir estrutura existente
- Testar antes de submeter
- Atualizar README se necessário

---

## 📚 Documentação Adicional

- **[SCALING_PLAN.md](docs/SCALING_PLAN.md)**: Arquitetura completa e estratégia de scaling
- **[NEXT_STEPS.md](docs/NEXT_STEPS.md)**: Guia de implementação fase-a-fase
- **[QLORA.MD](QLORA.MD)**: Guia de quantização e otimização de memória

---

## 📖 Referências e Recursos

### Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

### Frameworks e Bibliotecas
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)

### Tutoriais Relacionados
- [Building a RAG System from Scratch](https://www.langchain.com/tutorials)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [LLM Quantization Guide](https://huggingface.co/docs/transformers/main_classes/quantization)

---

## 📝 Licença

Este projeto é de uso **educacional** e foi desenvolvido para demonstração de técnicas de RAG e NLP.

---

## 👨‍💻 Autor

**Matheus Masago**

- 🌐 GitHub: [@matheusIACreator](https://github.com/matheusIACreator)
- 💼 LinkedIn: [Matheus Masago](https://www.linkedin.com/in/matheus-masago/)
- 📧 Email: [contato via LinkedIn]

---

## 🙏 Agradecimentos

Este projeto foi desenvolvido com apoio de:

- **[LangChain](https://www.langchain.com/)** - Framework RAG
- **[Hugging Face](https://huggingface.co/)** - Modelos e Embeddings
- **[ChromaDB](https://www.trychroma.com/)** - Vector Database
- **[Meta AI](https://ai.meta.com/)** - Llama Models
- **[Anthropic](https://www.anthropic.com/)** - Claude AI (development assistance)
- **[Wikipedia](https://www.wikipedia.org/)** - Fonte de dados históricos

---

## 🌟 Star History

Se este projeto foi útil para você, considere dar uma ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=matheusIACreator/Langchain_Project&type=Date)](https://star-history.com/#matheusIACreator/Langchain_Project&Date)

---

## 📞 Contato e Suporte

- **Issues**: [GitHub Issues](https://github.com/matheusIACreator/Langchain_Project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/matheusIACreator/Langchain_Project/discussions)
- **Email**: Disponível via LinkedIn

---

<div align="center">

**⚡ Construído com Python, LangChain, e muita ☕**

[⬆ Voltar ao topo](#-historical-figures-rag-system)

</div>

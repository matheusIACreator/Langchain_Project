# 🚀 Próximos Passos - Implementação Prática

## 📋 Status Atual

✅ **Já Implementado:**
- Sistema RAG básico funcional (Galileu)
- Vector store com ChromaDB
- Embeddings com Sentence Transformers
- LLM com Llama-3.1-8B-Instruct + QLoRA
- Memória conversacional
- Interface Gradio

📦 **Recém Criado:**
- `SCALING_PLAN.md` - Plano completo de arquitetura
- `config/experts_config.py` - Configuração de experts
- `src/retrieval/topic_router.py` - Sistema de roteamento
- `src/retrieval/hybrid_retriever.py` - Retrieval híbrido

---

## 🎯 Roadmap de Implementação

### **Fase 0: Preparação (Você faz agora)**

#### Decisão de Dados

**Opção A: Usar Wikipedia** (mais rápido)
- ✅ Vantagem: Dados já disponíveis
- ⚠️ Desvantagem: Qualidade variável, pode ter gaps

**Opção B: Criar PDFs curados** (mais qualidade)
- ✅ Vantagem: Controle total sobre conteúdo
- ✅ Vantagem: Melhor precisão histórica
- ⚠️ Desvantagem: Mais trabalho inicial

#### Figuras Prioritárias (escolha 3 para começar)

**Sugestão 1: Três Gigantes da Física**
1. Galileo Galilei (✅ já implementado) - Renascimento
2. Isaac Newton - Iluminismo
3. Albert Einstein - Era Moderna

**Sugestão 2: Períodos Diversos**
1. Leonardo da Vinci - Arte/Ciência Renascentista
2. Charles Darwin - Biologia/Evolução
3. Marie Curie - Física/Química Moderna

#### Estrutura de Dados Esperada

```
data/raw/
├── renaissance/
│   ├── galileo_galilei.pdf         (✅ já existe)
│   ├── leonardo_da_vinci.pdf       (📝 criar/obter)
│   └── michelangelo.pdf            (📝 criar/obter)
├── enlightenment/
│   ├── isaac_newton.pdf            (📝 criar/obter)
│   ├── voltaire.pdf                (📝 criar/obter)
│   └── benjamin_franklin.pdf       (📝 criar/obter)
└── modern_era/
    ├── albert_einstein.pdf         (📝 criar/obter)
    ├── marie_curie.pdf             (📝 criar/obter)
    └── charles_darwin.pdf          (📝 criar/obter)
```

---

### **Fase 1: Multi-Collection Vector Store** (1-2 semanas)

#### 1.1 Modificar `src/vectorstore.py`

**Objetivo:** Suportar múltiplas collections

```python
# Atual: Single collection
vectorstore = Chroma(
    collection_name="galileu_collection",
    ...
)

# Novo: Multi-collection
collections = {
    'renaissance/galileo_galilei': Chroma(...),
    'enlightenment/isaac_newton': Chroma(...),
    'modern_era/albert_einstein': Chroma(...),
}
```

**Arquivos a modificar:**
- `src/vectorstore.py` → `src/vectorstore_multi.py`
- Adicionar `MultiCollectionVectorStore` class

#### 1.2 Criar Pipeline de Ingestão

**Objetivo:** Automatizar processamento de novos PDFs

**Novo arquivo:** `src/ingestion/pipeline.py`

```python
class IngestionPipeline:
    def ingest_figure(self, pdf_path: str, period: str, figure_name: str):
        """
        Processa PDF de uma figura e cria collection
        
        Args:
            pdf_path: Caminho do PDF
            period: "renaissance", "enlightenment", etc
            figure_name: "galileo_galilei", "isaac_newton", etc
        """
        # 1. Carregar PDF
        # 2. Dividir em chunks
        # 3. Extrair metadados
        # 4. Criar/atualizar collection
        pass
```

**Teste:**
```bash
python src/ingestion/pipeline.py --pdf data/raw/enlightenment/isaac_newton.pdf --period enlightenment --figure isaac_newton
```

---

### **Fase 2: Integrar Topic Router** (1 semana)

#### 2.1 Integrar na RAG Chain

**Modificar:** `src/chains/rag_chain.py`

**Antes:**
```python
def query(self, question: str):
    docs = self.retriever.get_relevant_documents(question)
    # ...
```

**Depois:**
```python
def query(self, question: str):
    # 1. Rotear para expert apropriado
    routing = self.topic_router.route_query(question)
    
    # 2. Buscar nas collections relevantes
    collections = self.topic_router.route_to_collections(
        question, 
        self.available_periods
    )
    
    # 3. Retrieval multi-collection
    docs = self.multi_retriever.retrieve_from_collections(
        question, 
        collections
    )
    
    # 4. Gerar resposta
    # ...
```

---

### **Fase 3: Hybrid Retrieval** (1 semana)

#### 3.1 Instalar Dependência

```bash
pip install rank-bm25
```

#### 3.2 Integrar na RAG Chain

**Modificar:** `src/chains/rag_chain.py`

```python
from src.retrieval.hybrid_retriever import HybridRetriever

class GalileuRAGChain:
    def __init__(self):
        # ...
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.vectorstore.as_retriever(),
            documents=self.all_documents
        )
    
    def query(self, question: str):
        # Usar hybrid retrieval
        docs = self.hybrid_retriever.retrieve_hybrid(
            question, 
            k=TOP_K_DOCUMENTS
        )
        # ...
```

---

### **Fase 4: Testar com 3 Figuras** (1 semana)

#### 4.1 Processar Dados

```bash
# Galileu (já existe)
# Newton
python src/ingestion/pipeline.py --pdf data/raw/enlightenment/isaac_newton.pdf
# Einstein
python src/ingestion/pipeline.py --pdf data/raw/modern_era/albert_einstein.pdf
```

#### 4.2 Testar Queries Cross-Figure

```python
test_queries = [
    "Quando Galileu nasceu?",  # Single figure
    "Compare Newton e Einstein",  # Multi-figure
    "Como a física evoluiu do Renascimento à Era Moderna?",  # Multi-period
    "Quem descobriu as leis da gravidade?",  # Routing test
]
```

---

## 🎯 Decisão Imediata

**O que você precisa decidir agora:**

1. **Fonte de dados?**
   - [ ] Opção A: Usar Wikipedia (eu busco e processo)
   - [ ] Opção B: Você cria PDFs curados

2. **Quais 3 figuras começar?**
   - [ ] Galileu (✅ done), Newton, Einstein
   - [ ] Galileu (✅ done), Leonardo, Darwin
   - [ ] Outra combinação: ________________

3. **Próximo passo técnico?**
   - [ ] Implementar multi-collection vector store
   - [ ] Preparar dados primeiro
   - [ ] Implementar hybrid retrieval

---

## 📚 Datasets Wikipedia Disponíveis

Se escolher usar Wikipedia (Opção A), aqui estão os melhores datasets:

### 1. **wikimedia/wikipedia** (Recomendado)
- 77.4K downloads
- Mais atualizado e completo
- Link: https://hf.co/datasets/wikimedia/wikipedia

### 2. **BetterHF/wikipedia-biography-dataset**
- Específico para biografias
- Link: https://hf.co/datasets/BetterHF/wikipedia-biography-dataset

### 3. **Cohere/wikipedia-2023-11-embed-multilingual-v3**
- Com embeddings pré-computados
- Acelera busca inicial
- Link: https://hf.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3

**Script para baixar e processar:**
```python
from datasets import load_dataset

# Baixar biografias específicas
dataset = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train")

# Filtrar figuras de interesse
figures = ["Galileo_Galilei", "Isaac_Newton", "Albert_Einstein"]
for figure in figures:
    # Processar e salvar
    pass
```

---

## 🔧 Scripts Úteis para Começar

### Script 1: Download Wikipedia

```bash
# Criar script: scripts/download_wikipedia_figures.py
python scripts/download_wikipedia_figures.py --figures "Galileo,Newton,Einstein"
```

### Script 2: Convert Wikipedia to PDF

```bash
# Criar script: scripts/wikipedia_to_pdf.py
python scripts/wikipedia_to_pdf.py --figure "Isaac_Newton" --output data/raw/enlightenment/
```

### Script 3: Process All Figures

```bash
# Criar script: scripts/process_all_figures.py
python scripts/process_all_figures.py --data-dir data/raw/
```

---

## ✅ Checklist de Preparação

Antes de começar a implementação, complete:

- [ ] Decidir: Wikipedia ou PDFs curados
- [ ] Escolher 3 figuras prioritárias
- [ ] Se Wikipedia: baixar datasets relevantes
- [ ] Se PDFs: criar/obter PDFs das 3 figuras
- [ ] Organizar arquivos na estrutura `data/raw/periodo/figura.pdf`
- [ ] Instalar dependência: `pip install rank-bm25`
- [ ] Revisar `SCALING_PLAN.md` para entender arquitetura completa

---

## 💬 Me Avise Quando Estiver Pronto!

Depois que você:
1. Decidir sobre os dados (Wikipedia vs PDFs)
2. Escolher as 3 figuras
3. Ter os arquivos preparados (ou me pedir para baixar do Wikipedia)

**Eu implementarei:**
- Multi-collection vector store
- Pipeline de ingestão automatizada
- Integração do topic router
- Hybrid retrieval
- Testes end-to-end

---

## 🎓 Recursos Adicionais

### Tutoriais Criados:
- ✅ `SCALING_PLAN.md` - Arquitetura completa
- ✅ `config/experts_config.py` - Config de experts
- ✅ `src/retrieval/topic_router.py` - Sistema de routing
- ✅ `src/retrieval/hybrid_retriever.py` - Retrieval híbrido

### Próximos Tutoriais (após implementação):
- [ ] `docs/MULTI_COLLECTION_GUIDE.md`
- [ ] `docs/INGESTION_PIPELINE.md`
- [ ] `docs/MOE_IMPLEMENTATION.md`
- [ ] `docs/EVALUATION_METRICS.md`

---

**🚀 Pronto para começar! Me diga qual caminho quer seguir!**

# 🚀 Plano de Escalonamento - Sistema RAG Multi-Período Histórico

## 📋 Visão Geral

Transformar o chatbot atual (focado em Galileu Galilei) em um sistema escalável capaz de responder perguntas sobre múltiplas figuras históricas e períodos da humanidade.

---

## 🎯 Objetivos

1. **Escala Horizontal**: Suportar múltiplas figuras históricas (Galileu, Newton, Einstein, etc.)
2. **Escala Temporal**: Cobrir diferentes períodos históricos (Renascimento, Iluminismo, Revolução Científica, etc.)
3. **Especialização**: Modelos especializados por domínio usando Mixture of Experts (MoE)
4. **Eficiência**: Manter inferência eficiente com QLoRA
5. **Precisão Histórica**: Validação de fatos e citações como guardrails

---

## 🏗️ Arquitetura Proposta

### 1. Multi-Collection Vector Store (ChromaDB)

```
data/vectorstore/
├── renaissance/
│   ├── galileo_galilei/
│   ├── leonardo_da_vinci/
│   └── michelangelo/
├── enlightenment/
│   ├── isaac_newton/
│   ├── voltaire/
│   └── john_locke/
├── modern_physics/
│   ├── albert_einstein/
│   ├── marie_curie/
│   └── niels_bohr/
└── metadata/
    ├── temporal_index.json
    ├── thematic_index.json
    └── cross_references.json
```

**Vantagens**:
- Isolamento de contextos por período/figura
- Busca eficiente dentro de domínios específicos
- Facilita manutenção e atualização incremental

### 2. Topic-Based Routing System

```python
class TopicRouter:
    """
    Roteador inteligente que direciona queries para a collection apropriada
    """
    
    def __init__(self):
        self.classifiers = {
            'temporal': TemporalClassifier(),    # Século/período
            'thematic': ThematicClassifier(),    # Física/Astronomia/etc
            'entity': EntityRecognizer()         # Nome da figura
        }
    
    def route_query(self, query: str) -> List[str]:
        """
        Retorna lista de collections relevantes para a query
        
        Exemplo:
        Query: "Como Newton e Einstein viam a gravidade?"
        Return: ['enlightenment/isaac_newton', 'modern_physics/albert_einstein']
        """
        pass
```

### 3. Hybrid Retrieval (Dense + Sparse)

**Dense Retrieval** (já implementado):
- Embeddings semânticos (sentence-transformers)
- Captura relações conceituais
- Bom para queries abstratas

**Sparse Retrieval** (a implementar):
- BM25 ou TF-IDF
- Captura keywords exatos (nomes, datas, lugares)
- Bom para queries factuais

```python
class HybridRetriever:
    """
    Combina busca densa (semantic) e esparsa (keyword)
    """
    
    def __init__(self, vectorstore, bm25_index):
        self.dense_retriever = vectorstore.as_retriever()
        self.sparse_retriever = BM25Retriever(bm25_index)
    
    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Retorna top-k documentos combinando ambas as estratégias
        """
        dense_docs = self.dense_retriever.get_relevant_documents(query)
        sparse_docs = self.sparse_retriever.get_relevant_documents(query)
        
        # Reciprocal Rank Fusion (RRF) para combinar rankings
        return self._reciprocal_rank_fusion(dense_docs, sparse_docs, k)
```

---

## 🤖 Mixture of Experts (MoE) Architecture

### Conceito

Em vez de um único LLM generalista, usar múltiplos modelos especializados:

```
┌─────────────────────────────────────┐
│         Router LLM (pequeno)        │
│  "Qual especialista deve responder?" │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│   Physics    │  │  Biography  │  │  Philosophy │
│   Expert     │  │   Expert    │  │   Expert    │
│ (Fine-tuned) │  │ (Fine-tuned)│  │ (Fine-tuned)│
└──────────────┘  └─────────────┘  └─────────────┘
```

### Implementação com QLoRA

```python
class ExpertMoE:
    """
    Mixture of Experts com QLoRA para eficiência
    """
    
    def __init__(self):
        # Router leve para classificação rápida
        self.router = self._load_router()
        
        # Experts especializados (loaded on-demand)
        self.experts = {
            'physics': None,      # Llama fine-tuned em física
            'biography': None,    # Llama fine-tuned em biografias
            'philosophy': None,   # Llama fine-tuned em filosofia
        }
        
    def query(self, question: str, context: str) -> str:
        # 1. Router decide qual expert usar
        expert_type = self.router.classify(question)
        
        # 2. Carrega expert (lazy loading para economizar VRAM)
        if self.experts[expert_type] is None:
            self.experts[expert_type] = self._load_expert(expert_type)
        
        # 3. Expert gera resposta
        return self.experts[expert_type].generate(question, context)
    
    def _load_expert(self, expert_type: str):
        """
        Carrega expert com QLoRA 4-bit
        """
        return HuggingFacePipeline.from_pretrained(
            f"models/{expert_type}_expert",
            load_in_4bit=True,
            bnb_config=self.qlora_config
        )
```

**Vantagens**:
- Cada expert é menor e mais especializado
- QLoRA permite múltiplos experts em 4GB VRAM (load on-demand)
- Melhor qualidade em domínios específicos

---

## 📊 Pipeline de Ingestão Automatizada

### Estrutura de Metadados Rica

```python
class DocumentMetadata:
    """
    Metadados enriquecidos para cada chunk
    """
    
    # Identificação
    source_document: str
    chunk_id: int
    
    # Temporal
    period: str              # "Renaissance", "Enlightenment", etc.
    start_year: int          # 1564
    end_year: int            # 1642
    century: int             # 16, 17, 18, etc.
    
    # Geográfica
    primary_location: str    # "Italy", "England", etc.
    secondary_locations: List[str]
    
    # Temática
    main_topics: List[str]   # ["physics", "astronomy"]
    keywords: List[str]      # ["telescope", "jupiter", "moons"]
    
    # Entidades
    main_figure: str         # "Galileo Galilei"
    mentioned_figures: List[str]  # ["Copernicus", "Pope Urban VIII"]
    
    # Cross-referências
    related_chunks: List[int]
    contradicts_chunks: List[int]
```

### Pipeline Automatizado

```python
class DocumentIngestionPipeline:
    """
    Pipeline para processar e inserir novos documentos
    """
    
    def ingest_document(self, pdf_path: str, metadata: Dict):
        """
        Pipeline completo:
        1. Extração de texto
        2. Chunking inteligente
        3. Extração de metadados
        4. Enriquecimento (NER, temporal extraction)
        5. Cross-referencing
        6. Inserção no vector store
        """
        
        # 1. Extração
        text = self.pdf_loader.load(pdf_path)
        
        # 2. Chunking semântico (considera estrutura do documento)
        chunks = self.semantic_chunker.split(text)
        
        # 3. NER para extrair entidades
        for chunk in chunks:
            chunk.metadata.update({
                'mentioned_figures': self.ner.extract_persons(chunk.text),
                'locations': self.ner.extract_locations(chunk.text),
                'dates': self.temporal_extractor.extract_dates(chunk.text),
                'topics': self.topic_classifier.classify(chunk.text),
            })
        
        # 4. Cross-referencing
        self.cross_referencer.link_chunks(chunks)
        
        # 5. Inserção na collection apropriada
        collection_name = self._determine_collection(metadata)
        self.vectorstore.add_to_collection(collection_name, chunks)
```

---

## 🎓 Datasets Disponíveis

### 1. **Wikipedia** (wikimedia/wikipedia)
- **77.4K downloads**
- Multilingual (300+ idiomas)
- Biografias de figuras históricas
- **Uso**: Base principal para biografias e contexto histórico

### 2. **Wikipedia Biography Dataset** (BetterHF/wikipedia-biography-dataset)
- **127 downloads**
- Focado em biografias
- **Uso**: Treinamento de expert em biografias

### 3. **Wikipedia Embeddings** (Cohere/wikipedia-2023-11-embed-multilingual-v3)
- **12.5K downloads**
- Embeddings pré-computados
- **Uso**: Acelerar busca semântica

### 4. **RAG Mini Wikipedia** (rag-datasets/rag-mini-wikipedia)
- **3.3K downloads**
- Dataset específico para RAG
- **Uso**: Testes e validação

---

## 🛠️ Estrutura de Código Atualizada

```
Langchain_Project/
├── config/
│   ├── settings.py                 # Configurações gerais
│   └── experts_config.py           # Config dos experts MoE
├── data/
│   ├── raw/                        # PDFs organizados por período
│   │   ├── renaissance/
│   │   ├── enlightenment/
│   │   └── modern_era/
│   ├── processed/
│   └── vectorstore/                # Multi-collection
│       ├── renaissance/
│       ├── enlightenment/
│       └── modern_era/
├── src/
│   ├── ingestion/
│   │   ├── pdf_processor.py       # Processamento avançado de PDFs
│   │   ├── metadata_extractor.py  # NER, temporal extraction
│   │   └── pipeline.py             # Pipeline automatizado
│   ├── retrieval/
│   │   ├── hybrid_retriever.py    # Dense + Sparse
│   │   ├── topic_router.py        # Roteamento por tópico
│   │   └── cross_reference.py     # Sistema de cross-refs
│   ├── models/
│   │   ├── moe.py                 # Mixture of Experts
│   │   ├── expert_loader.py       # Carregamento de experts
│   │   └── router.py              # Router LLM
│   ├── evaluation/
│   │   ├── accuracy_metrics.py    # Métricas de precisão histórica
│   │   ├── citation_validator.py  # Validação de citações
│   │   └── benchmarks.py          # Benchmarks de performance
│   ├── chains/
│   │   └── multi_expert_chain.py  # Chain com MoE
│   └── utils/
│       ├── temporal_utils.py      # Utilities temporais
│       └── entity_utils.py        # Utilities de entidades
├── experts/                        # Modelos especializados
│   ├── physics_expert/
│   ├── biography_expert/
│   └── philosophy_expert/
└── tests/
    ├── test_routing.py
    ├── test_retrieval.py
    └── test_accuracy.py
```

---

## 📈 Métricas de Avaliação

### 1. Precisão Histórica

```python
class HistoricalAccuracyMetric:
    """
    Valida se as respostas contêm informações historicamente precisas
    """
    
    def evaluate(self, answer: str, ground_truth: Dict) -> float:
        """
        Verifica:
        - Datas corretas
        - Nomes corretos
        - Eventos na ordem certa
        - Sem anacronismos
        """
        score = 0.0
        
        # Validar datas
        if self._validate_dates(answer, ground_truth['dates']):
            score += 0.3
        
        # Validar nomes
        if self._validate_entities(answer, ground_truth['entities']):
            score += 0.3
        
        # Validar eventos
        if self._validate_timeline(answer, ground_truth['timeline']):
            score += 0.4
        
        return score
```

### 2. Validação de Citações

```python
class CitationValidator:
    """
    Verifica se as citações estão corretas e bem atribuídas
    """
    
    def validate(self, answer: str, sources: List[Document]) -> Dict:
        """
        Retorna:
        - Citações encontradas
        - Citações corretas
        - Citações sem fonte
        - Score de confiabilidade
        """
        pass
```

### 3. Coverage Metrics

```python
class CoverageMetrics:
    """
    Mede a cobertura do conhecimento histórico
    """
    
    def calculate_coverage(self) -> Dict:
        """
        Retorna:
        - Períodos cobertos
        - Figuras por período
        - Tópicos por figura
        - Gaps no conhecimento
        """
        pass
```

---

## 🚦 Roadmap de Implementação

### Fase 1: Preparação dos Dados (Semanas 1-2)
- [ ] Coletar/criar PDFs para figuras-chave
- [ ] Estruturar diretórios por período
- [ ] Implementar pipeline de ingestão automatizada
- [ ] Extrair metadados ricos

### Fase 2: Multi-Collection Vector Store (Semanas 3-4)
- [ ] Refatorar para suportar múltiplas collections
- [ ] Implementar topic-based routing
- [ ] Adicionar BM25 para hybrid retrieval
- [ ] Sistema de cross-referências

### Fase 3: Mixture of Experts (Semanas 5-7)
- [ ] Treinar/fine-tune experts especializados
- [ ] Implementar router LLM
- [ ] Sistema de lazy loading
- [ ] Testes de performance

### Fase 4: Avaliação e Guardrails (Semanas 8-9)
- [ ] Implementar métricas de precisão histórica
- [ ] Sistema de validação de citações
- [ ] Testes de cobertura
- [ ] Benchmarks comparativos

### Fase 5: Interface e Deploy (Semanas 10-12)
- [ ] Interface aprimorada (seletor de períodos)
- [ ] Visualizações temporais
- [ ] Sistema de feedback
- [ ] Documentação completa

---

## 💡 Figuras Prioritárias para Implementação

### Renascimento (1400-1600)
1. **Leonardo da Vinci** - Polímata
2. **Galileo Galilei** - Astronomia/Física (já implementado)
3. **Michelangelo** - Arte/Escultura

### Iluminismo (1650-1800)
1. **Isaac Newton** - Física/Matemática
2. **Voltaire** - Filosofia
3. **Benjamin Franklin** - Ciência/Política

### Era Moderna (1800-1950)
1. **Charles Darwin** - Biologia/Evolução
2. **Albert Einstein** - Física/Relatividade
3. **Marie Curie** - Química/Radioatividade

### Era Contemporânea (1950-)
1. **Richard Feynman** - Física Quântica
2. **Stephen Hawking** - Cosmologia
3. **Carl Sagan** - Astronomia/Divulgação

---

## 📦 Dependências Adicionais

```python
# requirements_scaling.txt

# Adicionar ao requirements.txt atual:

# NER e Processamento de Linguagem
spacy>=3.7.0
spacy-transformers>=1.3.0
# python -m spacy download pt_core_news_lg  # Modelo PT

# Sparse Retrieval
rank-bm25>=0.2.2

# Cross-referencing e Grafos
networkx>=3.2
pyvis>=0.3.2  # Visualização de grafos

# Extração de Entidades Temporais
dateparser>=1.2.0
arrow>=1.3.0

# Fine-tuning (se for fazer)
peft>=0.7.0  # Para LoRA
datasets>=2.16.0

# Métricas e Avaliação
evaluate>=0.4.1
rouge-score>=0.1.2
bert-score>=0.3.13

# Visualizações
plotly>=5.18.0
streamlit>=1.29.0  # Alternativa ao Gradio
```

---

## 🎯 Benefícios da Arquitetura

1. **Modularidade**: Cada componente pode ser atualizado independentemente
2. **Escalabilidade**: Adicionar nova figura = adicionar nova collection
3. **Eficiência**: QLoRA + lazy loading para rodar em 4GB VRAM
4. **Precisão**: Experts especializados + validação de fatos
5. **Manutenibilidade**: Código organizado por responsabilidade
6. **Extensibilidade**: Fácil adicionar novos períodos/figuras

---

## 🚨 Desafios e Considerações

### 1. Gestão de Memória
- **Problema**: Múltiplos experts podem exceder VRAM
- **Solução**: Lazy loading + offloading para CPU quando não em uso

### 2. Consistência Histórica
- **Problema**: Informações contraditórias entre fontes
- **Solução**: Sistema de votação + marcação de incerteza

### 3. Cross-Period Queries
- **Problema**: "Compare Newton e Einstein"
- **Solução**: Multi-collection retrieval + síntese especializada

### 4. Qualidade dos Dados
- **Problema**: PDFs podem ter erros ou ser tendenciosos
- **Solução**: Múltiplas fontes + validação cruzada

---

## 📚 Próximos Passos Imediatos

1. **Você fornecerá PDFs sobre outras figuras/períodos**
2. Implementaremos o pipeline de ingestão multi-collection
3. Desenvolveremos o sistema de routing
4. Testaremos com 2-3 figuras antes de escalar

---

## 🤝 Sugestão de Colaboração

Para maximizar eficiência, sugiro começarmos com:

1. **3 figuras piloto** de períodos diferentes:
   - Galileu (já implementado) - Renascimento
   - Newton - Iluminismo  
   - Einstein - Era Moderna

2. **Implementar primeiro**:
   - Multi-collection vector store
   - Topic routing básico
   - Hybrid retrieval

3. **Depois adicionar**:
   - Experts MoE
   - Validação de precisão
   - Mais figuras

---

## 📝 Conclusão

Esta arquitetura transforma seu chatbot de Galileu em uma plataforma robusta e escalável para explorar a história da ciência. O uso de:

- **Multi-collections** para organização
- **Hybrid retrieval** para precisão
- **MoE com QLoRA** para especialização eficiente
- **Rich metadata** para contexto
- **Validation metrics** para confiabilidade

... garante que o sistema pode crescer mantendo qualidade e performance.

**Pronto para começar? Podemos iniciar pela preparação dos dados ou pela implementação do multi-collection vector store!**

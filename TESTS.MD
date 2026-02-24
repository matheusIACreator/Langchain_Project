# 📑 Testes: RAG Multi-Figura v2.1

---

## 🎯 Motivação

O sistema v2.1 tinha todos os componentes implementados (Topic Router, Multi-Collection VectorStore, Hybrid Retriever, Memória e Chain) mas sem validação automatizada. A suite foi criada para prevenir regressões, permitir testes rápidos sem carregar o LLM e cobrir do componente isolado à integração total.

---

## 🏗️ As 4 Camadas de Teste

### 🟢 Camada 1: Topic Router
**17 testes | CPU apenas**

- Detecção de figuras (Galileu, Newton, Einstein) e queries comparativas
- Roteamento para categorias corretas (`biography`, `physics`, `astronomy`)
- Robustez: queries vazias, fallback e integridade do cache

### 🟡 Camada 2: Memória Conversacional
**9 testes | CPU apenas**

- Adição de mensagens e limpeza de memória
- Persistência de sessão via `save_load_estado`
- **Bug corrigido:** limite `k` só era aplicado no `add_user_message`. Corrigido centralizando a lógica no método `_enforce_window()`

### 🟠 Camada 3: Hybrid Retriever + VectorStore
**16 testes | ChromaDB ativo**

- Sanidade das collections (`renaissance`, `enlightenment`, `modern_era`)
- Isolamento: buscas sobre Einstein não retornam documentos de Galileu
- Multi-collection: queries comparativas trazem resultados de múltiplas fontes

### 🔴 Camada 4: Chain E2E
**17 testes | LLM + VectorStore**

- Detecção de cumprimentos, continuidade de memória com pronomes, estrutura do dicionário de saída
- **Status:** pendente — interrompido por timeout de download do modelo

---

## 🛠️ Decisões Técnicas

| Recurso | Benefício |
| :--- | :--- |
| **Fixtures (Module Scope)** | VectorStore criado uma vez por sessão, evita reindexar o BM25 a cada teste |
| **`pytest.skip`** | Distingue erro de infraestrutura de erro de lógica |
| **`@pytest.mark.llm`** | Permite `pytest -m "not llm"` para execução rápida sem GPU |
| **Hook `pytest_terminal_summary`** | Visibilidade clara do status de prontidão do sistema |

---

## 📈 Resultado Final

- **Executados:** 42/42 (Camadas 1, 2 e 3)
- **Sucesso:** 100% após correção do bug de memória
- **Pendência:** Camada 4 aguarda configuração de ambiente

> **Dica:** O modelo Llama-3.2-3B é o ideal para a RTX 3050 (4GB), garantindo que a Camada 4 rode sem estouro de VRAM.
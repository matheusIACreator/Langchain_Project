# 📑 Resumo de Testes: RAG Multi-Figura v2.0

---

## 🎯 Contexto e Motivação
A análise do estado real do projeto revelou que, embora o sistema v2.0 tivesse todos os componentes implementados (**Topic Router, Multi-Collection VectorStore, Hybrid Retriever, Memória e Chain**), não existia uma validação automatizada. 

A suite de testes foi criada para:
1. **Prevenir Regressões:** Garantir que novos bugs não cheguem a produção.
2. **Desenvolvimento Ágil:** Permitir testes rápidos sem a necessidade constante de carregar o LLM.
3. **Validação Progressiva:** Cobrir do componente isolado à integração total.

---

## 🏗️ Arquitetura dos Testes: As 4 Camadas
A suite organiza-se em quatro camadas independentes com requisitos de infraestrutura distintos:

### 🟢 Camada 1: Topic Router
* **Métricas:** 17 testes | **Requisitos:** CPU apenas (Sem GPU/LLM).
* **Foco:** O "cérebro" do roteamento.
    * **Detecção de Figuras:** Identifica nomes (Galileu, Newton, Einstein) e lida com queries comparativas.
    * **Roteamento de Experts:** Direciona perguntas para as categorias corretas (`biography`, `physics`, `astronomy`, `historical_context`).
    * **Robustez:** Tratamento de queries vazias, fallback para experts padrão e integridade do cache.

### 🟡 Camada 2: Memória Conversacional
* **Métricas:** 9 testes | **Requisitos:** CPU apenas.
* **Foco:** Gestão de contexto e janela de histórico.
    * **Operações Base:** Adição de mensagens e limpeza de memória.
    * **Serialização:** Teste de `save_load_estado` para garantir persistência de sessão.
    * **Bug Corrigido:** O teste `test_window_respeita_limite_k` revelou que o limite $k$ só era aplicado no `add_user_message`, deixando a resposta da IA extrapolar o limite. A lógica foi centralizada no método privado `_enforce_window()`.

### 🟠 Camada 3: Hybrid Retriever + VectorStore
* **Métricas:** 16 testes | **Requisitos:** VectorStore (ChromaDB) ativo.
* **Foco:** Integração entre busca vetorial e busca por palavra-chave.
    1. **Sanidade:** Verifica se as collections (`renaissance`, `enlightenment`, `modern_era`) existem e contêm documentos.
    2. **Isolamento:** Garante que buscas sobre Einstein não retornem documentos de Galileu (**Relevância Negativa**).
    3. **Multi-collection:** Valida se buscas comparativas trazem resultados de múltiplas fontes e respeitam o limite `k_final`.

### 🔴 Camada 4: Chain E2E Completa
* **Métricas:** 17 testes | **Requisitos:** LLM + VectorStore.
* **Status:** *Pendente de execução* (Interrompido por timeout de download do modelo).
* **Escopo:**
    * Detecção de cumprimentos (bypass de retrieval).
    * Continuidade de memória com pronomes (ex: "Quem foi Newton?" -> "Onde **ele** nasceu?").
    * Estrutura completa do dicionário de saída da Chain.

---

## 🛠️ Decisões Técnicas de Implementação

| Recurso | Descrição | Benefício |
| :--- | :--- | :--- |
| **Fixtures (Module Scope)** | VectorStore e Retriever criados uma única vez por sessão. | Evita reindexar o BM25 16 vezes, economizando tempo. |
| **`pytest.skip`** | Salta testes se o VectorStore estiver vazio. | Distingue "erro de infraestrutura" de "erro de lógica". |
| **Marcador `@pytest.mark.llm`** | Permite rodar `pytest -m "not llm"`. | Execução ultra-rápida (segundos) durante o coding. |
| **Hook Customizado** | `pytest_terminal_summary` formatado. | Visibilidade clara do status de prontidão do sistema. |

---

## 📈 Resultado Final da Sessão

* **Testes Executados:** 42/42 (Camadas 1, 2 e 3).
* **Sucesso:** 100% (após correção do bug de memória).
* **Bug Detectado:** Erro na lógica de truncagem da janela de contexto (Corrigido).
* **Pendência:** Camada 4 aguarda configuração de ambiente.

> [!TIP]
> O modelo de **3B** (Llama-3.2) é o ideal para a sua **RTX 3050 (4GB)**, garantindo que a Camada 4 rode sem estouro de VRAM ou timeouts excessivos.

---
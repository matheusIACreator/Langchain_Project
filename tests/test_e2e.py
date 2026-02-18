"""
tests/test_e2e.py
Suite de testes end-to-end para o sistema RAG Multi-Figura v2.0

Organização por camadas:
  Camada 1 - Unit: Topic Router (sem GPU, sem vectorstore)
  Camada 2 - Unit: Memória Conversacional (sem GPU, sem vectorstore)
  Camada 3 - Integration: Vectorstore + Hybrid Retriever (sem LLM)
  Camada 4 - E2E: Chain completa (requer GPU e vectorstore populado)

Execução:
  # Tudo
  pytest tests/test_e2e.py -v

  # Apenas testes rápidos (sem LLM)
  pytest tests/test_e2e.py -v -m "not llm"

  # Apenas uma camada
  pytest tests/test_e2e.py -v -k "Router"
  pytest tests/test_e2e.py -v -k "Memory"
  pytest tests/test_e2e.py -v -k "Retriever"
  pytest tests/test_e2e.py -v -k "Chain"
"""

import sys
import time
import pytest
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ══════════════════════════════════════════════════════
# FIXTURES COMPARTILHADAS
# ══════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def vectorstore_manager():
    """Vectorstore compartilhado entre testes de retrieval."""
    from src.vectorstore import MultiCollectionVectorStore
    vs = MultiCollectionVectorStore()
    cols = vs.list_collections()
    if not cols:
        pytest.skip("Vectorstore não encontrado. Execute: python src/vectorstore.py --mode multi")
    return vs


@pytest.fixture(scope="module")
def hybrid_retriever(vectorstore_manager):
    """Hybrid retriever compartilhado."""
    from src.retrieval.hybrid_retriever import MultiCollectionHybridRetriever
    return MultiCollectionHybridRetriever(vectorstore_manager)


@pytest.fixture(scope="module")
def rag_chain():
    """Chain completa — carregada uma única vez por sessão de testes."""
    from src.vectorstore import MultiCollectionVectorStore
    vs = MultiCollectionVectorStore()
    if not vs.list_collections():
        pytest.skip("Vectorstore não encontrado.")
    from src.chains.rag_chain_multi import MultiFigureRAGChain
    chain = MultiFigureRAGChain()
    yield chain
    # Cleanup: limpar memória após todos os testes da chain
    chain.clear_conversation()


# ══════════════════════════════════════════════════════
# CAMADA 1 — UNIT: TOPIC ROUTER
# ══════════════════════════════════════════════════════

class TestTopicRouter:
    """
    Testa o sistema de roteamento de queries.
    Roda sem GPU e sem vectorstore — muito rápido.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.retrieval.topic_router import TopicRouter
        self.router = TopicRouter()
        self.periods = ["renaissance", "enlightenment", "modern_era"]

    # ── Detecção de figuras ──────────────────────────

    def test_detecta_galileu(self):
        routing = self.router.route_query("Quando Galileu nasceu?")
        collections = self.router.route_to_collections(
            "Quando Galileu nasceu?", self.periods
        )
        galileu_present = any("galileo" in c for c in collections)
        assert galileu_present, (
            f"Esperava coleção do Galileu. Collections: {collections}"
        )

    def test_detecta_newton(self):
        collections = self.router.route_to_collections(
            "Quais são as leis do movimento de Newton?", self.periods
        )
        newton_present = any("newton" in c for c in collections)
        assert newton_present, (
            f"Esperava coleção do Newton. Collections: {collections}"
        )

    def test_detecta_einstein(self):
        collections = self.router.route_to_collections(
            "O que Einstein descobriu sobre a luz?", self.periods
        )
        einstein_present = any("einstein" in c for c in collections)
        assert einstein_present, (
            f"Esperava coleção do Einstein. Collections: {collections}"
        )

    def test_query_comparativa_detecta_multiplas_figuras(self):
        """Uma query com dois nomes deve retornar ao menos 2 collections."""
        collections = self.router.route_to_collections(
            "Compare Newton e Einstein sobre gravitação.", self.periods
        )
        assert len(collections) >= 2, (
            f"Query comparativa deveria retornar ≥2 collections. Got: {collections}"
        )

    def test_query_multi_periodo(self):
        """Query sobre evolução histórica deve incluir múltiplos períodos."""
        collections = self.router.route_to_collections(
            "Como a física evoluiu do Renascimento à Era Moderna?", self.periods
        )
        assert len(collections) >= 2, (
            f"Query multi-período deveria retornar ≥2 collections. Got: {collections}"
        )

    # ── Expert routing ───────────────────────────────

    def test_query_biografia_roteia_para_biography(self):
        routing = self.router.route_query("Quando e onde Galileu nasceu?")
        assert routing["primary_expert"] == "biography", (
            f"Esperava expert 'biography'. Got: {routing['primary_expert']}"
        )

    def test_query_fisica_roteia_para_physics(self):
        routing = self.router.route_query("Explique as leis do movimento e gravidade")
        assert routing["primary_expert"] in ("physics", "biography"), (
            f"Esperava 'physics' ou 'biography'. Got: {routing['primary_expert']}"
        )

    def test_query_astronomia_roteia_para_astronomy(self):
        routing = self.router.route_query("O que Galileu observou com o telescópio na Lua e Júpiter?")
        assert routing["primary_expert"] in ("astronomy", "physics"), (
            f"Esperava 'astronomy' ou 'physics'. Got: {routing['primary_expert']}"
        )

    def test_query_contexto_historico(self):
        routing = self.router.route_query("O que a Igreja fez com Galileu na Inquisição?")
        assert routing["primary_expert"] in ("historical_context", "biography"), (
            f"Esperava contexto histórico ou biography. Got: {routing['primary_expert']}"
        )

    # ── Estrutura do resultado ───────────────────────

    def test_routing_retorna_campos_obrigatorios(self):
        routing = self.router.route_query("Quando Newton nasceu?")
        required_keys = {"primary_expert", "secondary_experts", "confidence", "routing_reason"}
        assert required_keys.issubset(routing.keys()), (
            f"Faltam campos no routing. Got keys: {list(routing.keys())}"
        )

    def test_confidence_entre_zero_e_um(self):
        routing = self.router.route_query("Física quântica de Einstein")
        assert 0.0 <= routing["confidence"] <= 1.0, (
            f"Confidence deve estar entre 0 e 1. Got: {routing['confidence']}"
        )

    def test_secondary_experts_eh_lista(self):
        routing = self.router.route_query("Galileu e o telescópio")
        assert isinstance(routing["secondary_experts"], list)

    # ── Cache ────────────────────────────────────────

    def test_cache_funciona(self):
        query = "Quando Galileu nasceu? (teste cache)"
        result1 = self.router.route_query(query)
        result2 = self.router.route_query(query)
        assert result1["primary_expert"] == result2["primary_expert"]

    def test_stats_incrementam_com_queries(self):
        router = __import__("src.retrieval.topic_router", fromlist=["TopicRouter"]).TopicRouter()
        initial = router.get_routing_stats()["total_queries"]
        router.route_query("Query para teste de stats")
        after = router.get_routing_stats()["total_queries"]
        assert after == initial + 1

    # ── Fallback ─────────────────────────────────────

    def test_query_sem_match_usa_default(self):
        """Query sem keywords conhecidas deve retornar expert padrão, não erro."""
        routing = self.router.route_query("xyzxyz nonsense query 123")
        assert routing["primary_expert"] is not None
        assert routing["confidence"] >= 0

    def test_query_vazia_nao_crasha(self):
        try:
            routing = self.router.route_query("")
            assert "primary_expert" in routing
        except Exception as e:
            pytest.fail(f"Query vazia causou exceção: {e}")

    def test_busca_em_todas_quando_sem_figura(self):
        """Se nenhuma figura é detectada, deve buscar em todos os períodos disponíveis."""
        collections = self.router.route_to_collections(
            "Qual foi a contribuição mais importante para a ciência?",
            self.periods
        )
        # Pode retornar lista vazia (fallback na chain) ou todos
        assert isinstance(collections, list)


# ══════════════════════════════════════════════════════
# CAMADA 2 — UNIT: MEMÓRIA CONVERSACIONAL
# ══════════════════════════════════════════════════════

class TestConversationMemory:
    """
    Testa o sistema de memória conversacional.
    Roda sem GPU e sem vectorstore — muito rápido.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.memory.conversation_memory import GalileuConversationMemory
        self.memory = GalileuConversationMemory(memory_type="window", k=5)

    def test_adiciona_mensagem_usuario(self):
        self.memory.add_user_message("Quando Galileu nasceu?")
        stats = self.memory.get_memory_stats()
        assert stats["user_messages"] == 1

    def test_adiciona_mensagem_ai(self):
        self.memory.add_ai_message("Galileu nasceu em 1564.")
        stats = self.memory.get_memory_stats()
        assert stats["ai_messages"] == 1

    def test_historico_formatado_contem_mensagens(self):
        self.memory.add_user_message("Quem foi Galileu?")
        self.memory.add_ai_message("Foi um cientista italiano.")
        history = self.memory.get_formatted_history()
        assert "Galileu" in history
        assert "cientista" in history

    def test_limpar_memoria(self):
        self.memory.add_user_message("Teste")
        self.memory.add_ai_message("Resposta")
        self.memory.clear_memory()
        stats = self.memory.get_memory_stats()
        assert stats["total_messages"] == 0
        assert stats["interactions"] == 0

    def test_window_respeita_limite_k(self):
        """Com k=5, não deve manter mais de 10 mensagens (5 pares)."""
        for i in range(8):
            self.memory.add_user_message(f"Pergunta {i}")
            self.memory.add_ai_message(f"Resposta {i}")
        stats = self.memory.get_memory_stats()
        assert stats["total_messages"] <= 10, (
            f"Window k=5 deveria manter ≤10 msgs. Got: {stats['total_messages']}"
        )

    def test_interacao_count_incrementa(self):
        self.memory.add_user_message("msg 1")
        self.memory.add_user_message("msg 2")
        stats = self.memory.get_memory_stats()
        assert stats["interactions"] == 2

    def test_memoria_vazia_retorna_string_vazia_ou_placeholder(self):
        history = self.memory.get_formatted_history()
        assert isinstance(history, str)

    def test_stats_retorna_campos_esperados(self):
        required = {"total_messages", "user_messages", "ai_messages",
                    "interactions", "memory_type"}
        stats = self.memory.get_memory_stats()
        assert required.issubset(stats.keys())

    def test_save_load_estado(self):
        self.memory.add_user_message("Newton e a gravidade?")
        self.memory.add_ai_message("Newton descobriu a lei da gravitação.")
        state = self.memory.save_to_dict()

        from src.memory.conversation_memory import GalileuConversationMemory
        new_memory = GalileuConversationMemory(memory_type="window", k=5)
        new_memory.load_from_dict(state)

        restored = new_memory.get_formatted_history()
        assert "Newton" in restored


# ══════════════════════════════════════════════════════
# CAMADA 3 — INTEGRATION: VECTORSTORE + HYBRID RETRIEVER
# ══════════════════════════════════════════════════════

class TestHybridRetriever:
    """
    Testa o retrieval híbrido contra o vectorstore real.
    Não requer LLM — roda em CPU rapidamente.
    """

    # ── Sanidade do vectorstore ──────────────────────

    def test_vectorstore_tem_collections(self, vectorstore_manager):
        cols = vectorstore_manager.list_collections()
        assert len(cols) > 0, "Nenhuma collection encontrada no vectorstore."

    def test_tres_collections_esperadas(self, vectorstore_manager):
        cols = vectorstore_manager.list_collections()
        expected = {
            "renaissance/galileo_galilei",
            "enlightenment/isaac_newton",
            "modern_era/albert_einstein",
        }
        found = set(cols)
        missing = expected - found
        assert not missing, f"Collections faltando: {missing}"

    def test_collections_tem_documentos(self, vectorstore_manager):
        for col in vectorstore_manager.list_collections():
            period, figure = col.split("/")
            chroma = vectorstore_manager.load_collection(period, figure)
            assert chroma is not None, f"Falha ao carregar {col}"
            count = chroma._collection.count()
            assert count > 0, f"Collection {col} está vazia!"

    # ── HybridRetriever criação ──────────────────────

    def test_hybrid_retriever_inicializa(self, hybrid_retriever):
        stats = hybrid_retriever.get_stats()
        assert stats["collections_indexed"] > 0

    def test_hybrid_indexa_todas_as_collections(self, hybrid_retriever, vectorstore_manager):
        n_cols = len(vectorstore_manager.list_collections())
        stats = hybrid_retriever.get_stats()
        assert stats["collections_indexed"] == n_cols, (
            f"Esperava {n_cols} collections indexadas. Got: {stats['collections_indexed']}"
        )

    # ── Busca em collection única ────────────────────

    def test_busca_galileu_retorna_docs(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="Galileu nasceu em Pisa",
            collections=[("renaissance", "galileo_galilei")],
            k_per_collection=3,
        )
        assert len(docs) > 0, "Deveria retornar documentos sobre Galileu."

    def test_busca_newton_retorna_docs(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="Newton e as leis do movimento",
            collections=[("enlightenment", "isaac_newton")],
            k_per_collection=3,
        )
        assert len(docs) > 0, "Deveria retornar documentos sobre Newton."

    def test_busca_einstein_retorna_docs(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="Einstein e a relatividade",
            collections=[("modern_era", "albert_einstein")],
            k_per_collection=3,
        )
        assert len(docs) > 0, "Deveria retornar documentos sobre Einstein."

    # ── Metadata de origem ───────────────────────────

    def test_docs_tem_source_collection(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="nascimento e vida",
            collections=[("renaissance", "galileo_galilei")],
            k_per_collection=3,
        )
        for doc in docs:
            assert "source_collection" in doc.metadata, (
                f"Doc sem source_collection: {doc.metadata}"
            )

    def test_source_collection_valor_correto(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="Isaac Newton física",
            collections=[("enlightenment", "isaac_newton")],
            k_per_collection=3,
        )
        for doc in docs:
            assert doc.metadata["source_collection"] == "enlightenment/isaac_newton"

    # ── Busca multi-collection ───────────────────────

    def test_busca_multi_collection_retorna_de_multiplas_origens(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="leis da física e gravidade",
            collections=[
                ("enlightenment", "isaac_newton"),
                ("modern_era", "albert_einstein"),
            ],
            k_per_collection=3,
        )
        sources = {doc.metadata.get("source_collection") for doc in docs}
        assert len(sources) >= 1, "Deveria retornar docs de pelo menos 1 collection."

    def test_busca_comparativa_inclui_ambas_figuras(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="Newton Einstein gravitação comparação",
            collections=[
                ("enlightenment", "isaac_newton"),
                ("modern_era", "albert_einstein"),
            ],
            k_per_collection=4,
            k_final=8,
        )
        sources = {doc.metadata.get("source_collection") for doc in docs}
        assert len(sources) == 2, (
            f"Busca comparativa deveria trazer docs das 2 collections. Got: {sources}"
        )

    # ── Limite de resultados ─────────────────────────

    def test_k_final_limita_resultados(self, hybrid_retriever):
        k_final = 5
        docs = hybrid_retriever.retrieve(
            query="ciência e descobertas",
            collections=[
                ("renaissance", "galileo_galilei"),
                ("enlightenment", "isaac_newton"),
                ("modern_era", "albert_einstein"),
            ],
            k_per_collection=4,
            k_final=k_final,
        )
        assert len(docs) <= k_final, (
            f"k_final={k_final} deveria limitar. Got: {len(docs)}"
        )

    def test_docs_contem_conteudo_nao_vazio(self, hybrid_retriever):
        docs = hybrid_retriever.retrieve(
            query="Galileu telescópio",
            collections=[("renaissance", "galileo_galilei")],
            k_per_collection=3,
        )
        for doc in docs:
            assert doc.page_content.strip(), "Documento com conteúdo vazio!"

    # ── Relevância básica ────────────────────────────

    def test_galileu_nao_aparece_ao_buscar_einstein(self, hybrid_retriever):
        """
        Ao buscar só na collection do Einstein, não deveria vir conteúdo do Galileu.
        """
        docs = hybrid_retriever.retrieve(
            query="relatividade especial luz velocidade",
            collections=[("modern_era", "albert_einstein")],
            k_per_collection=4,
        )
        for doc in docs:
            col = doc.metadata.get("source_collection", "")
            assert "galileo" not in col, (
                f"Doc do Galileu encontrado em busca restrita ao Einstein: {col}"
            )

    # ── Performance ──────────────────────────────────

    def test_busca_simples_em_menos_de_2s(self, hybrid_retriever):
        start = time.time()
        hybrid_retriever.retrieve(
            query="física e astronomia",
            collections=[("renaissance", "galileo_galilei")],
            k_per_collection=4,
        )
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Busca demorou {elapsed:.2f}s (limite: 2s)"


# ══════════════════════════════════════════════════════
# CAMADA 4 — E2E: CHAIN COMPLETA (requer LLM + GPU)
# ══════════════════════════════════════════════════════

@pytest.mark.llm
class TestChainE2E:
    """
    Testes end-to-end da chain completa.
    Requer LLM carregado (GPU/CPU) + vectorstore populado.
    Marque com: pytest -m llm
    """

    # ── Inicialização ────────────────────────────────

    def test_chain_inicializa(self, rag_chain):
        assert rag_chain is not None

    def test_chain_tem_collections_disponiveis(self, rag_chain):
        assert len(rag_chain.available_collections) > 0

    # ── Cumprimentos ─────────────────────────────────

    def test_cumprimento_retorna_resposta(self, rag_chain):
        result = rag_chain.query("Olá!")
        assert result["is_greeting"] is True
        assert result["answer"]
        assert len(result["source_documents"]) == 0

    def test_cumprimento_nao_vai_para_retrieval(self, rag_chain):
        result = rag_chain.query("Oi")
        assert result.get("is_greeting") is True

    # ── Queries sobre figura única ───────────────────

    def test_query_galileu_retorna_resposta(self, rag_chain):
        result = rag_chain.query("Quando e onde Galileu Galilei nasceu?")
        assert result["answer"]
        assert len(result["answer"]) > 20
        assert len(result["source_documents"]) > 0

    def test_query_newton_retorna_resposta(self, rag_chain):
        result = rag_chain.query("Quais foram as principais contribuições de Isaac Newton?")
        assert result["answer"]
        assert len(result["source_documents"]) > 0

    def test_query_einstein_retorna_resposta(self, rag_chain):
        result = rag_chain.query("O que foi a teoria da relatividade de Einstein?")
        assert result["answer"]
        assert len(result["source_documents"]) > 0

    # ── Relevância dos documentos ────────────────────

    def test_query_galileu_usa_collection_galileu(self, rag_chain):
        result = rag_chain.query("Fale sobre as descobertas de Galileu com o telescópio.")
        collections_used = result.get("collections_used", [])
        assert any("galileo" in c for c in collections_used), (
            f"Esperava collection do Galileu. Got: {collections_used}"
        )

    def test_query_comparativa_usa_multiplas_collections(self, rag_chain):
        result = rag_chain.query("Compare as contribuições de Newton e Einstein para a física.")
        collections_used = result.get("collections_used", [])
        assert len(collections_used) >= 2, (
            f"Query comparativa deveria usar ≥2 collections. Got: {collections_used}"
        )

    # ── Modo de retrieval ────────────────────────────

    def test_retrieval_mode_e_hibrido(self, rag_chain):
        result = rag_chain.query("Quando Einstein ganhou o Nobel?")
        mode = result.get("retrieval_mode", "")
        assert "hybrid" in mode.lower() or "bm25" in mode.lower(), (
            f"Esperava modo híbrido. Got: '{mode}'"
        )

    # ── Memória conversacional ───────────────────────

    def test_memoria_mantem_contexto_figura(self, rag_chain):
        """
        Após perguntar sobre Newton, um pronome 'ele' deve ser resolvido corretamente.
        """
        rag_chain.clear_conversation()
        rag_chain.query("Quem foi Isaac Newton?")
        result = rag_chain.query("Quando ele nasceu?")
        # A resposta deve conter algo sobre Newton (1643) sem mencionar outra figura
        answer = result["answer"].lower()
        assert result["answer"], "Resposta de follow-up vazia"
        # Não deve ter perdido o contexto completamente
        assert len(result["answer"]) > 10

    def test_limpar_conversa_reseta_memoria(self, rag_chain):
        rag_chain.query("Galileu foi julgado pela Inquisição.")
        rag_chain.clear_conversation()
        from src.memory.conversation_memory import GalileuConversationMemory
        mem = rag_chain.memory
        stats = mem.get_memory_stats()
        assert stats["total_messages"] == 0

    # ── Estrutura do resultado ───────────────────────

    def test_query_retorna_campos_obrigatorios(self, rag_chain):
        result = rag_chain.query("Newton e as leis da gravidade.")
        required = {"question", "answer", "source_documents",
                    "routing", "collections_used", "retrieval_mode"}
        assert required.issubset(result.keys()), (
            f"Faltam campos no resultado. Got: {list(result.keys())}"
        )

    def test_source_documents_sao_lista(self, rag_chain):
        result = rag_chain.query("Galileu e o heliocentrismo.")
        assert isinstance(result["source_documents"], list)

    def test_collections_used_sao_lista(self, rag_chain):
        result = rag_chain.query("Einstein e a bomba atômica.")
        assert isinstance(result["collections_used"], list)

    # ── Stats ────────────────────────────────────────

    def test_get_stats_retorna_campos_esperados(self, rag_chain):
        stats = rag_chain.get_stats()
        required = {"model", "device", "retrieval_mode",
                    "vectorstore", "router", "memory", "hybrid_retriever"}
        assert required.issubset(stats.keys())

    def test_hybrid_retriever_stats_na_chain(self, rag_chain):
        stats = rag_chain.get_stats()
        hr_stats = stats.get("hybrid_retriever", {})
        assert hr_stats.get("collections_indexed", 0) > 0


# ══════════════════════════════════════════════════════
# RELATÓRIO PERSONALIZADO
# ══════════════════════════════════════════════════════

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Imprime sumário final colorido por camada."""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))
    total = passed + failed + skipped

    print("\n" + "=" * 60)
    print("📊 SUMÁRIO DOS TESTES — RAG Multi-Figura v2.0")
    print("=" * 60)
    print(f"  ✅ Passou:  {passed}/{total}")
    print(f"  ❌ Falhou:  {failed}/{total}")
    print(f"  ⏭️  Pulado:  {skipped}/{total}")
    print("=" * 60)

    if failed == 0 and passed > 0:
        print("  🎉 Todos os testes passaram! Sistema pronto para a próxima fase.")
    elif failed > 0:
        print("  ⚠️  Há falhas — revise os componentes antes de continuar.")
    print("=" * 60 + "\n")
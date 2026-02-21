"""
Camada 5 — Métricas de Avaliação do Retrieval
MRR (Mean Reciprocal Rank) e NDCG (Normalized Discounted Cumulative Gain)

Avalia a qualidade do Hybrid Retriever com um conjunto de queries
com ground truth definido manualmente.

Execução:
    pytest tests/test_retrieval_metrics.py -v
    pytest tests/test_retrieval_metrics.py -v --tb=short
"""

import math
import sys
import pytest
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════
# GROUND TRUTH
# Cada entrada define:
#   - query: pergunta de avaliação
#   - collection: collection esperada
#   - relevant_keywords: termos que DEVEM aparecer nos docs recuperados
#   - figure: figura esperada nos metadados
# ══════════════════════════════════════════════════════
GROUND_TRUTH: List[Dict] = [
    # ── Galileu ───────────────────────────────────────
    {
        "query": "Quando e onde Galileu nasceu?",
        "collection": "renaissance/galileo_galilei",
        "figure": "galileo_galilei",
        "relevant_keywords": ["1564", "pisa", "galileo", "february"],
    },
    {
        "query": "Galileu e o telescópio",
        "collection": "renaissance/galileo_galilei",
        "figure": "galileo_galilei",
        "relevant_keywords": ["telescope", "galileo", "observations", "lens"],
    },
    {
        "query": "Galileu e a Inquisição",
        "collection": "renaissance/galileo_galilei",
        "figure": "galileo_galilei",
        "relevant_keywords": ["heliocentrism", "pope", "urban", "advocate", "dialogue", "geocentric"],
    },
    {
        "query": "Galileu e as luas de Júpiter",
        "collection": "renaissance/galileo_galilei",
        "figure": "galileo_galilei",
        "relevant_keywords": ["jupiter", "moons", "satellites", "galileo"],
    },
    {
        "query": "queda livre galileu experimentos física",
        "collection": "renaissance/galileo_galilei",
        "figure": "galileo_galilei",
        "relevant_keywords": ["fall", "motion", "physics", "experiment", "velocity"],
    },
    # ── Newton ────────────────────────────────────────
    {
        "query": "Quando Newton nasceu?",
        "collection": "enlightenment/isaac_newton",
        "figure": "isaac_newton",
        "relevant_keywords": ["1643", "newton", "woolsthorpe", "cambridge"],
    },
    {
        "query": "Newton e a lei da gravitação universal",
        "collection": "enlightenment/isaac_newton",
        "figure": "isaac_newton",
        "relevant_keywords": ["gravitation", "gravity", "newton", "law"],
    },
    {
        "query": "Newton e o cálculo matemático",
        "collection": "enlightenment/isaac_newton",
        "figure": "isaac_newton",
        "relevant_keywords": ["calculus", "leibniz", "mathematics", "fluxions"],
    },
    {
        "query": "Principia Mathematica Newton",
        "collection": "enlightenment/isaac_newton",
        "figure": "isaac_newton",
        "relevant_keywords": ["principia", "philosophiae", "naturalis", "mathematica"],
    },
    {
        "query": "Newton óptica luz prisma cores",
        "collection": "enlightenment/isaac_newton",
        "figure": "isaac_newton",
        "relevant_keywords": ["optics", "light", "prism", "colour", "spectrum"],
    },
    # ── Einstein ──────────────────────────────────────
    {
        "query": "Quando Einstein nasceu?",
        "collection": "modern_era/albert_einstein",
        "figure": "albert_einstein",
        "relevant_keywords": ["1879", "einstein", "ulm", "germany", "born"],
    },
    {
        "query": "Einstein e a teoria da relatividade",
        "collection": "modern_era/albert_einstein",
        "figure": "albert_einstein",
        "relevant_keywords": ["relativity", "einstein", "space", "time", "special"],
    },
    {
        "query": "Einstein Nobel de Física",
        "collection": "modern_era/albert_einstein",
        "figure": "albert_einstein",
        "relevant_keywords": ["nobel", "einstein", "1921", "prize", "physics"],
    },
    {
        "query": "E=mc² energia massa Einstein",
        "collection": "modern_era/albert_einstein",
        "figure": "albert_einstein",
        "relevant_keywords": ["energy", "mass", "equivalence", "light", "speed"],
    },
    {
        "query": "Einstein efeito fotoelétrico quantum",
        "collection": "modern_era/albert_einstein",
        "figure": "albert_einstein",
        "relevant_keywords": ["photoelectric", "quantum", "photon", "light", "effect"],
    },
]
# ══════════════════════════════════════════════════════
# FUNÇÕES DE MÉTRICAS
# ══════════════════════════════════════════════════════

def is_relevant(doc, ground_truth: Dict) -> bool:
    """
    Determina se um documento é relevante para uma query.

    Um documento é considerado relevante se:
    1. Pertence à collection esperada (via metadados), E
    2. Contém pelo menos um keyword relevante no conteúdo
    """
    content = doc.page_content.lower()
    source = doc.metadata.get("source_collection", "").lower()

    # Verificar collection
    expected_collection = ground_truth["collection"].lower()
    collection_match = expected_collection in source or source in expected_collection

    # Verificar keywords
    keywords = ground_truth["relevant_keywords"]
    keyword_match = any(kw.lower() in content for kw in keywords)

    return collection_match and keyword_match


def reciprocal_rank(docs: List, ground_truth: Dict) -> float:
    """
    Calcula o Reciprocal Rank para uma query.

    RR = 1 / posição_do_primeiro_doc_relevante
    Se nenhum doc é relevante → RR = 0
    """
    for rank, doc in enumerate(docs, start=1):
        if is_relevant(doc, ground_truth):
            return 1.0 / rank
    return 0.0


def dcg(docs: List, ground_truth: Dict, k: int) -> float:
    """
    Discounted Cumulative Gain até posição k.

    DCG@k = Σ rel_i / log2(i + 1)
    rel_i = 1 se doc na posição i é relevante, 0 caso contrário
    """
    score = 0.0
    for i, doc in enumerate(docs[:k], start=1):
        relevance = 1.0 if is_relevant(doc, ground_truth) else 0.0
        score += relevance / math.log2(i + 1)
    return score


def ideal_dcg(ground_truth: Dict, k: int) -> float:
    """
    IDCG — DCG ideal assumindo que todos os docs relevantes
    aparecem nas primeiras posições.

    Para o nosso caso binário (relevante/não relevante),
    o IDCG@k = Σ 1/log2(i+1) para i=1..min(num_relevantes, k)
    Assumimos que há pelo menos 1 documento relevante.
    """
    # Estimativa conservadora: assumimos que existem pelo menos
    # k documentos relevantes no corpus (já que temos 96-127 por collection)
    num_relevant = k  # assume que todos os k docs poderiam ser relevantes
    return sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))


def ndcg(docs: List, ground_truth: Dict, k: int) -> float:
    """
    Normalized DCG até posição k.

    NDCG@k = DCG@k / IDCG@k
    Retorna valor entre 0 e 1.
    """
    idcg = ideal_dcg(ground_truth, k)
    if idcg == 0:
        return 0.0
    return dcg(docs, ground_truth, k) / idcg


# ══════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def hybrid_retriever():
    """Inicializa o retriever híbrido uma vez para todos os testes."""
    from src.vectorstore import MultiCollectionVectorStore
    from src.retrieval.hybrid_retriever import MultiCollectionHybridRetriever

    vs_dir = Path("data/vectorstore")
    if not vs_dir.exists() or not any(vs_dir.rglob("*.sqlite3")):
        pytest.skip("Vectorstore não encontrado. Execute: python src/vectorstore.py --mode multi")

    vs = MultiCollectionVectorStore()
    retriever = MultiCollectionHybridRetriever(vs)
    return retriever


@pytest.fixture(scope="module")
def topic_router():
    """Inicializa o Topic Router uma vez para todos os testes."""
    from src.retrieval.topic_router import TopicRouter
    return TopicRouter()


# ══════════════════════════════════════════════════════
# CAMADA 5 — TESTES DE MÉTRICAS
# ══════════════════════════════════════════════════════

class TestRetrievalMetrics:
    """
    Avalia a qualidade do Hybrid Retriever com métricas MRR e NDCG.

    Os testes individuais validam cada query do ground truth.
    O teste de sumário calcula as médias globais e imprime o relatório.
    """

    K = 8  # número de documentos recuperados por query

    # ── Testes por query ──────────────────────────────

    @pytest.mark.parametrize("gt", GROUND_TRUTH, ids=[g["query"][:40] for g in GROUND_TRUTH])
    def test_retrieval_retorna_docs_relevantes(self, hybrid_retriever, gt):
        """Cada query deve retornar pelo menos 1 documento relevante."""
        period, figure = gt["collection"].split("/")
        docs = hybrid_retriever.retrieve(
            query=gt["query"],
            collections=[(period, figure)],
            k_per_collection=self.K,
            k_final=self.K,
        )
        assert len(docs) > 0, f"Nenhum documento retornado para: {gt['query']}"

        has_relevant = any(is_relevant(doc, gt) for doc in docs)
        assert has_relevant, (
            f"Nenhum doc relevante para: '{gt['query']}'\n"
            f"  Esperado: {gt['collection']} + keywords {gt['relevant_keywords']}\n"
            f"  Recebido: {[doc.metadata.get('source_collection') for doc in docs]}"
        )

    @pytest.mark.parametrize("gt", GROUND_TRUTH, ids=[g["query"][:40] for g in GROUND_TRUTH])
    def test_mrr_por_query_aceitavel(self, hybrid_retriever, gt):
        """RR de cada query deve ser ≥ 0.25 (doc relevante nas top 4 posições)."""
        period, figure = gt["collection"].split("/")
        docs = hybrid_retriever.retrieve(
            query=gt["query"],
            collections=[(period, figure)],
            k_per_collection=self.K,
            k_final=self.K,
        )
        rr = reciprocal_rank(docs, gt)
        assert rr >= 0.25, (
            f"RR baixo para '{gt['query']}': {rr:.3f}\n"
            f"  Doc relevante não encontrado nas top 4 posições."
        )

    @pytest.mark.parametrize("gt", GROUND_TRUTH, ids=[g["query"][:40] for g in GROUND_TRUTH])
    def test_ndcg_por_query_aceitavel(self, hybrid_retriever, gt):
        """NDCG@8 de cada query deve ser ≥ 0.3."""
        period, figure = gt["collection"].split("/")
        docs = hybrid_retriever.retrieve(
            query=gt["query"],
            collections=[(period, figure)],
            k_per_collection=self.K,
            k_final=self.K,
        )
        score = ndcg(docs, gt, k=self.K)
        assert score >= 0.3, (
            f"NDCG@{self.K} baixo para '{gt['query']}': {score:.3f}"
        )

    # ── Sumário global ────────────────────────────────

    def test_mrr_global_aceitavel(self, hybrid_retriever):
        """MRR médio sobre todas as queries deve ser ≥ 0.5."""
        rrs = []
        for gt in GROUND_TRUTH:
            period, figure = gt["collection"].split("/")
            docs = hybrid_retriever.retrieve(
                query=gt["query"],
                collections=[(period, figure)],
                k_per_collection=self.K,
                k_final=self.K,
            )
            rrs.append(reciprocal_rank(docs, gt))

        mrr = sum(rrs) / len(rrs)

        print(f"\n{'='*60}")
        print(f"📊 MÉTRICAS DE RETRIEVAL — RAG Multi-Figura v2.0")
        print(f"{'='*60}")
        print(f"  Queries avaliadas : {len(GROUND_TRUTH)}")
        print(f"  K (top-k)         : {self.K}")
        print(f"  MRR               : {mrr:.4f}  (threshold ≥ 0.50)")

        # Por figura
        for figure_key in ["galileo_galilei", "isaac_newton", "albert_einstein"]:
            figure_gts = [g for g in GROUND_TRUTH if g["figure"] == figure_key]
            figure_rrs = []
            for gt in figure_gts:
                period, figure = gt["collection"].split("/")
                docs = hybrid_retriever.retrieve(
                    query=gt["query"],
                    collections=[(period, figure)],
                    k_per_collection=self.K,
                    k_final=self.K,
                )
                figure_rrs.append(reciprocal_rank(docs, gt))
            fig_mrr = sum(figure_rrs) / len(figure_rrs) if figure_rrs else 0
            label = figure_key.replace("_", " ").title()
            print(f"    {label:<20}: MRR = {fig_mrr:.4f}")

        print(f"{'='*60}\n")

        assert mrr >= 0.5, f"MRR global insuficiente: {mrr:.4f} (mínimo: 0.50)"

    def test_ndcg_global_aceitavel(self, hybrid_retriever):
        """NDCG@8 médio sobre todas as queries deve ser ≥ 0.5."""
        scores = []
        for gt in GROUND_TRUTH:
            period, figure = gt["collection"].split("/")
            docs = hybrid_retriever.retrieve(
                query=gt["query"],
                collections=[(period, figure)],
                k_per_collection=self.K,
                k_final=self.K,
            )
            scores.append(ndcg(docs, gt, k=self.K))

        mean_ndcg = sum(scores) / len(scores)

        print(f"\n  NDCG@{self.K}           : {mean_ndcg:.4f}  (threshold ≥ 0.50)")

        # Por figura
        for figure_key in ["galileo_galilei", "isaac_newton", "albert_einstein"]:
            figure_gts = [g for g in GROUND_TRUTH if g["figure"] == figure_key]
            figure_scores = []
            for gt in figure_gts:
                period, figure = gt["collection"].split("/")
                docs = hybrid_retriever.retrieve(
                    query=gt["query"],
                    collections=[(period, figure)],
                    k_per_collection=self.K,
                    k_final=self.K,
                )
                figure_scores.append(ndcg(docs, gt, k=self.K))
            fig_ndcg = sum(figure_scores) / len(figure_scores) if figure_scores else 0
            label = figure_key.replace("_", " ").title()
            print(f"    {label:<20}: NDCG@{self.K} = {fig_ndcg:.4f}")

        assert mean_ndcg >= 0.5, f"NDCG@{self.K} global insuficiente: {mean_ndcg:.4f} (mínimo: 0.50)"

    def test_colecao_correta_nos_resultados(self, hybrid_retriever):
        """
        Para cada query, todos os docs devem vir da collection correcta.
        Valida o isolamento entre figuras.
        """
        violacoes = []
        for gt in GROUND_TRUTH:
            period, figure = gt["collection"].split("/")
            docs = hybrid_retriever.retrieve(
                query=gt["query"],
                collections=[(period, figure)],
                k_per_collection=self.K,
                k_final=self.K,
            )
            for doc in docs:
                source = doc.metadata.get("source_collection", "")
                if gt["collection"] not in source and source not in gt["collection"]:
                    violacoes.append(
                        f"Query '{gt['query'][:30]}': doc de '{source}' em vez de '{gt['collection']}'"
                    )

        assert len(violacoes) == 0, (
            f"Isolamento violado em {len(violacoes)} casos:\n" +
            "\n".join(f"  - {v}" for v in violacoes[:5])
        )
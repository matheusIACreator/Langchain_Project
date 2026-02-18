"""
Hybrid Retriever v2.0 - Busca híbrida multi-collection
Combina busca densa (semântica) e esparsa (BM25) por collection,
fundindo os rankings com Reciprocal Rank Fusion (RRF).
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

from config.settings import TOP_K_DOCUMENTS, DEBUG


class HybridRetriever:
    """
    Retriever híbrido para uma única collection.
    Combina busca semântica (dense) e BM25 (sparse) via RRF.
    """

    def __init__(self, dense_retriever, documents: List[Document] = None):
        """
        Args:
            dense_retriever: Retriever do vectorstore Chroma (busca semântica).
            documents: Documentos para indexar no BM25.
        """
        print("🔍 Inicializando Hybrid Retriever...")

        self.dense_retriever = dense_retriever
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_documents: List[Document] = []

        if documents:
            self._build_bm25_index(documents)

        print("✅ Hybrid Retriever inicializado!")

    # ──────────────────────────────────────────
    # Indexação BM25
    # ──────────────────────────────────────────

    def _build_bm25_index(self, documents: List[Document]):
        """Constrói índice BM25 a partir dos documentos."""
        print(f"📚 Construindo índice BM25 ({len(documents)} documentos)...")
        tokenized = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized)
        self.bm25_documents = documents
        print("✅ Índice BM25 criado!")

    def add_documents(self, documents: List[Document]):
        """Adiciona documentos ao índice BM25 (rebuild completo)."""
        all_docs = self.bm25_documents + documents
        self._build_bm25_index(all_docs)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenização simples para BM25."""
        return text.lower().split()

    # ──────────────────────────────────────────
    # Retrieval individual
    # ──────────────────────────────────────────

    def retrieve_dense(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Busca semântica usando embeddings."""
        # invoke() substitui o depreciado get_relevant_documents()
        docs = self.dense_retriever.invoke(query)[:k]
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]

    def retrieve_sparse(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Busca por keywords com BM25."""
        if self.bm25_index is None:
            print("⚠️  BM25 index não construído — retornando lista vazia.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]

        return [
            (self.bm25_documents[i], float(scores[i]))
            for i in top_indices
            if float(scores[i]) > 0  # ignorar documentos sem match
        ]

    # ──────────────────────────────────────────
    # Hybrid retrieval principal
    # ──────────────────────────────────────────

    def retrieve_hybrid(
        self,
        query: str,
        k: int = TOP_K_DOCUMENTS,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Document]:
        """
        Busca híbrida: dense + sparse → Reciprocal Rank Fusion.

        Args:
            query: Pergunta do usuário.
            k: Número de documentos finais.
            dense_weight: Peso da busca semântica (padrão 0.6).
            sparse_weight: Peso do BM25 (padrão 0.4).

        Returns:
            Lista de documentos ranqueados.
        """
        dense_results = self.retrieve_dense(query, k=k * 2)
        sparse_results = self.retrieve_sparse(query, k=k * 2)

        if DEBUG:
            print(f"   Dense: {len(dense_results)} | Sparse: {len(sparse_results)}")

        return self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    # ──────────────────────────────────────────
    # Reciprocal Rank Fusion
    # ──────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        k: int = TOP_K_DOCUMENTS,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        k_constant: int = 60,
    ) -> List[Document]:
        """
        Combina dois rankings usando RRF.
        Score RRF = Σ weight / (k_constant + rank)
        """
        doc_scores: Dict[str, Dict] = {}

        for rank, (doc, _) in enumerate(dense_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = dense_weight / (k_constant + rank)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += rrf_score

        for rank, (doc, _) in enumerate(sparse_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = sparse_weight / (k_constant + rank)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += rrf_score

        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        return [item["doc"] for item in sorted_docs[:k]]

    def _get_doc_id(self, doc: Document) -> str:
        """ID único por documento (chunk_id ou hash do conteúdo)."""
        chunk_id = doc.metadata.get("chunk_id")
        source = doc.metadata.get("source", "")
        if chunk_id is not None:
            return f"{source}::{chunk_id}"
        return str(hash(doc.page_content))

    # ──────────────────────────────────────────
    # Debug / Estatísticas
    # ──────────────────────────────────────────

    def get_retrieval_stats(self, query: str, k: int = TOP_K_DOCUMENTS) -> Dict:
        """Retorna estatísticas comparando os métodos de retrieval."""
        dense_results = self.retrieve_dense(query, k=k)
        sparse_results = self.retrieve_sparse(query, k=k)
        hybrid_results = self.retrieve_hybrid(query, k=k)

        dense_ids = {self._get_doc_id(doc) for doc, _ in dense_results}
        sparse_ids = {self._get_doc_id(doc) for doc, _ in sparse_results}

        overlap = len(dense_ids & sparse_ids)

        return {
            "query": query,
            "k": k,
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "hybrid_count": len(hybrid_results),
            "overlap_dense_sparse": overlap,
            "overlap_percentage": (overlap / k * 100) if k > 0 else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MultiCollectionHybridRetriever
#  Orquestra HybridRetriever por collection, mantendo índices BM25 separados.
# ══════════════════════════════════════════════════════════════════════════════

class MultiCollectionHybridRetriever:
    """
    Hybrid Retriever para múltiplas collections ChromaDB.

    - Mantém um HybridRetriever (com índice BM25) por collection.
    - Carrega os documentos do ChromaDB para alimentar o BM25.
    - Executa busca híbrida por collection e faz RRF global no final.
    """

    def __init__(self, vectorstore_manager):
        """
        Args:
            vectorstore_manager: Instância de MultiCollectionVectorStore.
        """
        print("\n🔍 Inicializando Multi-Collection Hybrid Retriever...")

        self.vectorstore_manager = vectorstore_manager
        # {collection_name: HybridRetriever}
        self._retrievers: Dict[str, HybridRetriever] = {}

        # Pré-carregar índices para collections já disponíveis
        for col_name in vectorstore_manager.list_collections():
            self._ensure_retriever(col_name)

        loaded = len(self._retrievers)
        print(f"✅ Multi-Collection Hybrid Retriever pronto! ({loaded} collections indexadas)")

    # ──────────────────────────────────────────
    # Criação lazy de retrievers por collection
    # ──────────────────────────────────────────

    def _ensure_retriever(self, collection_name: str) -> Optional[HybridRetriever]:
        """
        Garante que exista um HybridRetriever para a collection.
        Cria na primeira chamada (lazy).
        """
        if collection_name in self._retrievers:
            return self._retrievers[collection_name]

        print(f"\n🔧 Construindo retriever híbrido para: {collection_name}")

        # Obter Chroma para esta collection
        parts = collection_name.split("/")
        if len(parts) != 2:
            print(f"⚠️  Formato inválido: {collection_name}")
            return None

        period, figure = parts
        chroma = self.vectorstore_manager.load_collection(period, figure)

        if chroma is None:
            print(f"⚠️  Collection não encontrada: {collection_name}")
            return None

        # Carregar todos os documentos da collection para o BM25
        documents = self._load_all_documents(chroma, collection_name)

        if not documents:
            print(f"⚠️  Nenhum documento encontrado em: {collection_name}")
            return None

        # Criar retriever dense a partir do Chroma
        dense_retriever = chroma.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_DOCUMENTS * 2},
        )

        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            documents=documents,
        )

        self._retrievers[collection_name] = retriever
        return retriever

    def _load_all_documents(
        self, chroma, collection_name: str
    ) -> List[Document]:
        """
        Carrega todos os documentos do ChromaDB para alimentar o BM25.
        Usa a API interna do Chroma para recuperar sem query.
        """
        try:
            # Chroma permite buscar N documentos; usamos um número alto
            # para pegar tudo (padrão máximo do Chroma é 10k por chamada)
            result = chroma._collection.get(
                include=["documents", "metadatas"]
            )

            docs = []
            for content, metadata in zip(
                result["documents"], result["metadatas"]
            ):
                metadata = metadata or {}
                metadata["source_collection"] = collection_name
                docs.append(Document(page_content=content, metadata=metadata))

            print(f"   📄 {len(docs)} documentos carregados de {collection_name}")
            return docs

        except Exception as e:
            print(f"⚠️  Erro ao carregar documentos de {collection_name}: {e}")
            return []

    # ──────────────────────────────────────────
    # Busca principal
    # ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        collections: List[Tuple[str, str]],
        k_per_collection: int = TOP_K_DOCUMENTS,
        k_final: int = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Document]:
        """
        Busca híbrida em múltiplas collections com RRF global.

        Args:
            query: Pergunta do usuário.
            collections: Lista de (period, figure) a consultar.
            k_per_collection: Docs a recuperar por collection antes da fusão.
            k_final: Total de docs no resultado final (default = k_per_collection * 2).
            dense_weight: Peso da busca densa.
            sparse_weight: Peso da busca BM25.

        Returns:
            Lista de documentos ranqueados por relevância global.
        """
        if k_final is None:
            k_final = k_per_collection * 2

        if DEBUG:
            print(f"\n🔍 Hybrid retrieval em {len(collections)} collections")
            print(f"   Query: {query[:60]}...")

        all_results_with_scores: List[Tuple[Document, float]] = []

        for period, figure in collections:
            col_name = f"{period}/{figure}"
            retriever = self._ensure_retriever(col_name)

            if retriever is None:
                continue

            # Busca híbrida nesta collection
            docs = retriever.retrieve_hybrid(
                query,
                k=k_per_collection,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )

            # Adicionar metadado de origem
            for doc in docs:
                doc.metadata["source_collection"] = col_name

            # Score decrescente por posição (para o RRF global)
            for rank, doc in enumerate(docs, 1):
                all_results_with_scores.append((doc, 1.0 / rank))

            if DEBUG:
                print(f"   ✓ {col_name}: {len(docs)} docs")

        if not all_results_with_scores:
            return []

        # RRF global entre todas as collections
        final_docs = self._global_rrf(all_results_with_scores, k=k_final)

        if DEBUG:
            print(f"   Total após RRF global: {len(final_docs)} docs")

        return final_docs

    def _global_rrf(
        self,
        results: List[Tuple[Document, float]],
        k: int,
        k_constant: int = 60,
    ) -> List[Document]:
        """RRF global entre resultados de múltiplas collections."""
        doc_scores: Dict[str, Dict] = {}

        for rank, (doc, _) in enumerate(results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (k_constant + rank)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += rrf_score

        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        return [item["doc"] for item in sorted_docs[:k]]

    def _get_doc_id(self, doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        source = doc.metadata.get("source", "")
        col = doc.metadata.get("source_collection", "")
        if chunk_id is not None:
            return f"{col}::{source}::{chunk_id}"
        return str(hash(doc.page_content))

    # ──────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Estatísticas dos retrievers carregados."""
        return {
            "collections_indexed": len(self._retrievers),
            "collections": list(self._retrievers.keys()),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Teste standalone
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🧪 TESTANDO MULTI-COLLECTION HYBRID RETRIEVER")
    print("=" * 60)

    try:
        from src.vectorstore import MultiCollectionVectorStore

        vs = MultiCollectionVectorStore()
        collections = vs.list_collections()

        if not collections:
            print("⚠️  Nenhuma collection encontrada.")
            print("   Execute: python src/vectorstore.py --mode multi")
            return

        print(f"\n📦 Collections disponíveis: {collections}")

        retriever = MultiCollectionHybridRetriever(vs)

        test_queries = [
            "Quando Galileu nasceu?",
            "Newton e as leis do movimento",
            "Compare Einstein e Newton sobre gravidade",
        ]

        # Todas as collections para os testes comparativos
        all_cols = [tuple(c.split("/")) for c in collections]

        for query in test_queries:
            print(f"\n{'─' * 60}")
            print(f"Query: {query}")
            results = retriever.retrieve(query, collections=all_cols, k_per_collection=3)
            print(f"Resultados: {len(results)} documentos")
            for i, doc in enumerate(results[:3], 1):
                col = doc.metadata.get("source_collection", "?")
                print(f"  {i}. [{col}] {doc.page_content[:80]}...")

        print("\n✅ Teste concluído!")

    except Exception as e:
        import traceback
        print(f"\n❌ ERRO: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
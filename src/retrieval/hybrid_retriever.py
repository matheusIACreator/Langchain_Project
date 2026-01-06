"""
Hybrid Retriever - Combina busca densa (semantic) e esparsa (BM25)
Melhora a precisão da recuperação usando ambas as estratégias
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    """
    Retriever híbrido que combina busca densa e esparsa
    """
    
    def __init__(self, dense_retriever, documents: List[Document] = None):
        """
        Inicializa o hybrid retriever
        
        Args:
            dense_retriever: Retriever do vectorstore (busca semântica)
            documents: Documentos para indexar no BM25 (opcional)
        """
        print("🔍 Inicializando Hybrid Retriever...")
        
        self.dense_retriever = dense_retriever
        self.bm25_index = None
        self.bm25_documents = []
        
        if documents:
            self._build_bm25_index(documents)
        
        print("✅ Hybrid Retriever inicializado!")
    
    def _build_bm25_index(self, documents: List[Document]):
        """
        Constrói índice BM25 para busca esparsa
        
        Args:
            documents: Documentos a indexar
        """
        print(f"📚 Construindo índice BM25 para {len(documents)} documentos...")
        
        # Tokenizar documentos
        tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        
        # Criar índice BM25
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_documents = documents
        
        print(f"✅ Índice BM25 criado!")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokeniza texto para BM25
        
        Args:
            text: Texto a tokenizar
            
        Returns:
            Lista de tokens
        """
        # Tokenização simples (pode ser melhorada com spaCy)
        return text.lower().split()
    
    def add_documents(self, documents: List[Document]):
        """
        Adiciona documentos ao índice BM25
        
        Args:
            documents: Novos documentos
        """
        if self.bm25_index is None:
            self._build_bm25_index(documents)
        else:
            # Rebuild index com novos documentos
            all_docs = self.bm25_documents + documents
            self._build_bm25_index(all_docs)
    
    def retrieve_dense(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Busca usando embeddings semânticos (dense)
        
        Args:
            query: Query de busca
            k: Número de documentos a retornar
            
        Returns:
            Lista de (documento, score)
        """
        # Busca semântica
        docs = self.dense_retriever.get_relevant_documents(query)[:k]
        
        # Retornar com scores (assumindo score uniforme se não disponível)
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]
    
    def retrieve_sparse(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Busca usando BM25 (sparse)
        
        Args:
            query: Query de busca
            k: Número de documentos a retornar
            
        Returns:
            Lista de (documento, score)
        """
        if self.bm25_index is None:
            print("⚠️  BM25 index não foi construído!")
            return []
        
        # Tokenizar query
        tokenized_query = self._tokenize(query)
        
        # Buscar com BM25
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Pegar top-k
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        results = [
            (self.bm25_documents[i], float(scores[i]))
            for i in top_k_indices
        ]
        
        return results
    
    def retrieve_hybrid(
        self,
        query: str,
        k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[Document]:
        """
        Busca híbrida combinando dense e sparse
        
        Args:
            query: Query de busca
            k: Número de documentos finais
            dense_weight: Peso para busca densa (0-1)
            sparse_weight: Peso para busca esparsa (0-1)
            
        Returns:
            Lista de documentos ranqueados
        """
        # Buscar com ambos os métodos (k*2 para ter mais opções)
        dense_results = self.retrieve_dense(query, k=k*2)
        sparse_results = self.retrieve_sparse(query, k=k*2)
        
        # Combinar usando Reciprocal Rank Fusion (RRF)
        combined_docs = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        return combined_docs
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        k_constant: int = 60
    ) -> List[Document]:
        """
        Combina rankings usando Reciprocal Rank Fusion
        
        RRF score para cada documento = Σ 1/(k + rank_i)
        onde rank_i é a posição em cada ranking
        
        Args:
            dense_results: Resultados da busca densa
            sparse_results: Resultados da busca esparsa
            k: Número de documentos finais
            dense_weight: Peso para rankings dense
            sparse_weight: Peso para rankings sparse
            k_constant: Constante RRF (padrão 60)
            
        Returns:
            Lista de documentos combinados e ranqueados
        """
        # Mapear documentos para scores
        doc_scores = {}
        
        # Processar dense results
        for rank, (doc, _) in enumerate(dense_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = dense_weight / (k_constant + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0.0}
            doc_scores[doc_id]['score'] += rrf_score
        
        # Processar sparse results
        for rank, (doc, _) in enumerate(sparse_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = sparse_weight / (k_constant + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0.0}
            doc_scores[doc_id]['score'] += rrf_score
        
        # Ordenar por score combinado
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Retornar top-k documentos
        return [item['doc'] for item in sorted_docs[:k]]
    
    def _get_doc_id(self, doc: Document) -> str:
        """
        Gera ID único para um documento
        
        Args:
            doc: Documento
            
        Returns:
            ID único
        """
        # Usar chunk_id se disponível, senão usar hash do conteúdo
        chunk_id = doc.metadata.get('chunk_id')
        if chunk_id is not None:
            return str(chunk_id)
        
        # Fallback: hash do conteúdo
        return str(hash(doc.page_content))
    
    def get_retrieval_stats(self, query: str, k: int = 10) -> Dict:
        """
        Retorna estatísticas comparando os métodos de retrieval
        
        Args:
            query: Query de teste
            k: Número de documentos
            
        Returns:
            Dict com estatísticas
        """
        # Buscar com cada método
        dense_results = self.retrieve_dense(query, k=k)
        sparse_results = self.retrieve_sparse(query, k=k)
        hybrid_results = self.retrieve_hybrid(query, k=k)
        
        # Calcular overlap
        dense_ids = {self._get_doc_id(doc) for doc, _ in dense_results}
        sparse_ids = {self._get_doc_id(doc) for doc, _ in sparse_results}
        hybrid_ids = {self._get_doc_id(doc) for doc in hybrid_results}
        
        overlap_dense_sparse = len(dense_ids & sparse_ids)
        
        stats = {
            'query': query,
            'k': k,
            'dense_results': len(dense_results),
            'sparse_results': len(sparse_results),
            'hybrid_results': len(hybrid_results),
            'overlap_dense_sparse': overlap_dense_sparse,
            'overlap_percentage': (overlap_dense_sparse / k) * 100 if k > 0 else 0,
        }
        
        return stats


def main():
    """
    Função principal para teste standalone
    """
    print("\n" + "="*60)
    print("🧪 TESTANDO HYBRID RETRIEVER")
    print("="*60 + "\n")
    
    # Criar documentos de teste
    test_docs = [
        Document(
            page_content="Galileu Galilei nasceu em 15 de fevereiro de 1564 em Pisa, Itália.",
            metadata={'chunk_id': 0, 'page': 1}
        ),
        Document(
            page_content="Em 1609, Galileu construiu um telescópio e observou as luas de Júpiter.",
            metadata={'chunk_id': 1, 'page': 2}
        ),
        Document(
            page_content="Isaac Newton nasceu em 1643 e desenvolveu as três leis do movimento.",
            metadata={'chunk_id': 2, 'page': 1}
        ),
        Document(
            page_content="Einstein propôs a teoria da relatividade em 1905.",
            metadata={'chunk_id': 3, 'page': 1}
        ),
    ]
    
    # Criar mock dense retriever
    class MockDenseRetriever:
        def __init__(self, docs):
            self.docs = docs
        
        def get_relevant_documents(self, query: str):
            # Simples: retorna todos os docs
            return self.docs
    
    dense_retriever = MockDenseRetriever(test_docs)
    
    # Inicializar hybrid retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        documents=test_docs
    )
    
    # Testar busca
    test_queries = [
        "Quando Galileu nasceu?",
        "O que Galileu descobriu com o telescópio?",
        "Quem foi Isaac Newton?",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}\n")
        
        # Busca híbrida
        results = hybrid_retriever.retrieve_hybrid(query, k=3)
        
        print("📚 Resultados híbridos:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc.page_content[:80]}...")
        
        # Estatísticas
        stats = hybrid_retriever.get_retrieval_stats(query, k=3)
        print(f"\n📊 Estatísticas:")
        print(f"   Overlap dense/sparse: {stats['overlap_percentage']:.1f}%")
    
    print("\n✅ Teste concluído!")


if __name__ == "__main__":
    main()

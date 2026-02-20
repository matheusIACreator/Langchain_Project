"""
Multi-Figure RAG Chain v2.1
Integra Topic Router + Multi-Collection Hybrid Retriever (BM25 + dense) + LLM
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from config.settings import (
    MODEL_NAME,
    MODEL_KWARGS,
    GENERATION_KWARGS,
    TOP_K_DOCUMENTS,
    HF_TOKEN,
    DEVICE,
    DEBUG,
)
from src.vectorstore import MultiCollectionVectorStore
from src.memory.conversation_memory import GalileuConversationMemory
from src.retrieval.topic_router import TopicRouter
from src.retrieval.hybrid_retriever import MultiCollectionHybridRetriever
from src.prompts.rag_prompts import (
    RAG_PROMPT,
    format_docs,
    GREETING_RESPONSES,
    OUT_OF_SCOPE_RESPONSE,
)


class MultiFigureRAGChain:
    """
    RAG Chain para múltiplas figuras históricas.

    Pipeline:
        query → TopicRouter → MultiCollectionHybridRetriever → LLM → resposta
    """

    def __init__(self):
        print("\n" + "=" * 60)
        print("🚀 INICIALIZANDO MULTI-FIGURE RAG CHAIN v2.1")
        print("=" * 60 + "\n")

        # 1. Vector Store
        self.vectorstore_manager = MultiCollectionVectorStore()

        # 2. Topic Router
        self.topic_router = TopicRouter()

        # 3. Hybrid Retriever (substitui busca semântica pura)
        print("\n⚙️  Configurando Hybrid Retriever...")
        self.hybrid_retriever = MultiCollectionHybridRetriever(
            self.vectorstore_manager
        )

        # 4. Memória conversacional
        self.memory = GalileuConversationMemory(memory_type="window", k=5)

        # 5. LLM
        self.llm = self._setup_llm()

        # 6. Mapear collections disponíveis
        self.available_collections = self._parse_collections()
        self._print_available_figures()

        print("\n✅ Multi-Figure RAG Chain v2.1 pronta!")

    # ──────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────

    def _setup_llm(self) -> HuggingFacePipeline:
        """Configura o LLM (Llama com quantização)."""
        print(f"\n🤖 Carregando LLM: {MODEL_NAME}")
        print(f"   Device: {DEVICE}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, token=HF_TOKEN, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                trust_remote_code=True,
                **MODEL_KWARGS,
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **GENERATION_KWARGS,
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            print("✅ LLM carregado!")
            return llm

        except Exception as e:
            print(f"❌ Erro ao carregar LLM: {e}")
            raise

    def _parse_collections(self) -> Dict[str, List[str]]:
        """Parseia collections em estrutura {period: [figures]}."""
        collections = self.vectorstore_manager.list_collections()
        parsed: Dict[str, List[str]] = {}
        for col in collections:
            parts = col.split("/")
            if len(parts) == 2:
                period, figure = parts
                parsed.setdefault(period, []).append(figure)
        return parsed

    def _print_available_figures(self):
        """Imprime figuras disponíveis no terminal."""
        print("\n📚 Figuras Disponíveis:")
        for period, figures in self.available_collections.items():
            print(f"  📁 {period.upper()}:")
            for fig in figures:
                print(f"     • {fig.replace('_', ' ').title()}")

    # ──────────────────────────────────────────
    # Detecção de intent
    # ──────────────────────────────────────────

    def _is_greeting(self, query: str) -> bool:
        greetings = ["oi", "olá", "ola", "hey", "opa", "bom dia", "boa tarde", "boa noite"]
        q = query.lower().strip()
        return any(g in q for g in greetings) and len(q.split()) <= 3

    # ──────────────────────────────────────────
    # Retrieval (agora com HybridRetriever)
    # ──────────────────────────────────────────

    def _resolve_collections(self, routing: Dict) -> List[Tuple[str, str]]:
        """
        Converte o resultado do TopicRouter em lista de (period, figure).
        Fallback: todas as collections disponíveis.
        """
        raw_collections = self.topic_router.route_to_collections(
            routing.get("query", ""),
            list(self.available_collections.keys()),
        )

        if not raw_collections:
            # Fallback: buscar em tudo
            return [
                (period, figure)
                for period, figures in self.available_collections.items()
                for figure in figures
            ]

        tuples = []
        for col_spec in raw_collections:
            if "/" in col_spec:
                period, figure = col_spec.split("/", 1)
                tuples.append((period, figure))
            elif col_spec in self.available_collections:
                # É um período — adicionar todas as figuras do período
                for figure in self.available_collections[col_spec]:
                    tuples.append((col_spec, figure))

        return tuples if tuples else [
            (period, figure)
            for period, figures in self.available_collections.items()
            for figure in figures
        ]

    def _retrieve_documents(self, query: str, routing: Dict) -> List[Document]:
        """
        Recupera documentos usando Hybrid Retrieval (BM25 + semântico + RRF).
        """
        collections = self._resolve_collections({**routing, "query": query})

        if DEBUG:
            print(f"\n🔍 Retrieval Híbrido:")
            print(f"   Expert: {routing.get('primary_expert', 'N/A')}")
            print(f"   Confiança: {routing.get('confidence', 0):.0%}")
            print(f"   Collections: {[f'{p}/{f}' for p, f in collections]}")

        docs = self.hybrid_retriever.retrieve(
            query=query,
            collections=collections,
            k_per_collection=TOP_K_DOCUMENTS,
            k_final=TOP_K_DOCUMENTS * 2,
        )

        return docs

    # ──────────────────────────────────────────
    # Geração de resposta
    # ──────────────────────────────────────────

    def _generate_answer(self, query: str, context: str, chat_history: str) -> str:
        """Gera resposta usando o LLM."""
        prompt = RAG_PROMPT.format(
            context=context,
            chat_history=chat_history,
            question=query,
        )

        if DEBUG:
            print("\n📝 Gerando resposta...")

        try:
            response = self.llm.invoke(prompt)
            # Extrair apenas a resposta gerada (após o marcador do prompt)
            answer = response.split("**Sua resposta:**")[-1].strip()
            return answer
        except Exception as e:
            return f"Desculpe, erro ao gerar resposta: {e}"

    # ──────────────────────────────────────────
    # Interface pública
    # ──────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """
        Processa uma query completa.

        Returns:
            Dict com: question, answer, source_documents,
                      routing, collections_used, retrieval_mode
        """
        print(f"\n{'=' * 60}")
        print(f"💬 Query: {question}")
        print(f"{'=' * 60}")

        # Cumprimentos
        if self._is_greeting(question):
            import random
            answer = random.choice(GREETING_RESPONSES)
            return {
                "question": question,
                "answer": answer,
                "source_documents": [],
                "is_greeting": True,
                "retrieval_mode": "none",
            }

        # Memória
        self.memory.add_user_message(question)

        # Roteamento
        routing = self.topic_router.route_query(question)

        # Retrieval híbrido
        docs = self._retrieve_documents(question, routing)

        # Contexto e histórico
        context = format_docs(docs)
        chat_history = self.memory.get_formatted_history()

        # Geração
        answer = self._generate_answer(question, context, chat_history)

        # Salvar na memória
        self.memory.add_ai_message(answer)

        result = {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "routing": routing,
            "retrieval_mode": "hybrid (dense + BM25 + RRF)",
            "collections_used": list(
                set(doc.metadata.get("source_collection", "unknown") for doc in docs)
            ),
        }

        if DEBUG:
            print(f"\n✅ Resposta gerada!")
            print(f"   Collections usadas: {result['collections_used']}")
            print(f"   Modo: {result['retrieval_mode']}")

        return result

    def chat(self, message: str) -> str:
        """Interface simplificada — retorna apenas a resposta."""
        return self.query(message)["answer"]

    def clear_conversation(self):
        """Limpa histórico da conversa."""
        self.memory.clear_memory()
        print("🗑️  Conversa reiniciada!")

    def get_stats(self) -> Dict:
        """Retorna estatísticas completas do sistema."""
        vs_stats = self.vectorstore_manager.get_stats()
        router_stats = self.topic_router.get_routing_stats()
        memory_stats = self.memory.get_memory_stats()
        hybrid_stats = self.hybrid_retriever.get_stats()

        return {
            "model": MODEL_NAME,
            "device": DEVICE,
            "retrieval_mode": "hybrid (dense + BM25 + RRF)",
            "vectorstore": vs_stats,
            "router": router_stats,
            "memory": memory_stats,
            "hybrid_retriever": hybrid_stats,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Teste standalone
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🧪 TESTANDO MULTI-FIGURE RAG CHAIN v2.1")
    print("=" * 60)

    try:
        chain = MultiFigureRAGChain()

        test_queries = [
            "Quando Galileu nasceu?",
            "Quais foram as leis de Newton?",
            "O que Einstein descobriu sobre a luz?",
            "Compare Newton e Einstein sobre gravitação.",
            "Como a física evoluiu do Renascimento à Era Moderna?",
        ]

        for query in test_queries:
            print(f"\n{'─' * 60}")
            result = chain.query(query)
            print(f"💬 Pergunta: {result['question']}")
            print(f"🤖 Resposta: {result['answer'][:200]}...")
            print(f"📚 Docs usados: {len(result['source_documents'])}")
            print(f"📦 Collections: {result.get('collections_used', [])}")
            print(f"🔍 Modo: {result.get('retrieval_mode', 'N/A')}")

        # Stats finais
        print(f"\n{'=' * 60}")
        print("📊 ESTATÍSTICAS DO SISTEMA")
        print(f"{'=' * 60}")
        stats = chain.get_stats()
        print(f"Modelo: {stats['model']}")
        print(f"Device: {stats['device']}")
        print(f"Modo retrieval: {stats['retrieval_mode']}")
        print(f"Collections indexadas: {stats['hybrid_retriever']['collections_indexed']}")

        print("\n✅ Teste concluído com sucesso!")

    except Exception as e:
        import traceback
        print(f"\n❌ ERRO: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
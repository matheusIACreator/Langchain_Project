"""
Multi-Figure RAG Chain
Integra Topic Router + Multi-Collection VectorStore + LLM
para responder perguntas sobre múltiplas figuras históricas
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

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
    DEBUG
)
from src.vectorstore import MultiCollectionVectorStore
from src.memory.conversation_memory import GalileuConversationMemory
from src.retrieval.topic_router import TopicRouter
from src.prompts.rag_prompts import RAG_PROMPT, format_docs, GREETING_RESPONSES, OUT_OF_SCOPE_RESPONSE


class MultiFigureRAGChain:
    """
    RAG Chain para múltiplas figuras históricas
    Usa Topic Router para direcionar queries inteligentemente
    """
    
    def __init__(self):
        """
        Inicializa o Multi-Figure RAG Chain
        """
        print("\n" + "="*60)
        print("🚀 INICIALIZANDO MULTI-FIGURE RAG CHAIN")
        print("="*60 + "\n")
        
        # Componentes
        self.vectorstore_manager = MultiCollectionVectorStore()
        self.topic_router = TopicRouter()
        self.memory = GalileuConversationMemory(memory_type="window", k=5)
        self.llm = self._setup_llm()
        
        # Descobrir collections disponíveis
        self.available_collections = self._parse_collections()
        
        print("\n✅ Multi-Figure RAG Chain inicializada!")
        self._print_available_figures()
    
    def _parse_collections(self) -> Dict[str, List[str]]:
        """
        Parseia collections em estrutura period -> [figures]
        
        Returns:
            Dict mapeando período -> lista de figuras
        """
        collections = self.vectorstore_manager.list_collections()
        
        parsed = {}
        for col in collections:
            period, figure = col.split('/')
            if period not in parsed:
                parsed[period] = []
            parsed[period].append(figure)
        
        return parsed
    
    def _print_available_figures(self):
        """Imprime figuras disponíveis"""
        print("\n📚 Figuras Disponíveis:")
        for period, figures in self.available_collections.items():
            print(f"\n  📁 {period.upper()}:")
            for figure in figures:
                print(f"     • {figure.replace('_', ' ').title()}")
    
    def _setup_llm(self) -> HuggingFacePipeline:
        """Configura o LLM"""
        print(f"\n🤖 Carregando LLM: {MODEL_NAME}")
        print(f"   Device: {DEVICE}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                trust_remote_code=True,
                **MODEL_KWARGS
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **GENERATION_KWARGS
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            
            print(f"✅ LLM carregado!")
            return llm
            
        except Exception as e:
            print(f"❌ Erro ao carregar LLM: {str(e)}")
            raise
    
    def _is_greeting(self, query: str) -> bool:
        """Detecta cumprimentos"""
        greetings = ["oi", "olá", "ola", "hey", "opa", "bom dia", "boa tarde", "boa noite"]
        query_lower = query.lower().strip()
        return any(g in query_lower for g in greetings) and len(query_lower.split()) <= 3
    
    def _retrieve_documents(self, query: str, routing: Dict) -> List[Document]:
        """
        Recupera documentos usando routing inteligente
        
        Args:
            query: Query do usuário
            routing: Informações de routing
            
        Returns:
            Lista de documentos relevantes
        """
        if DEBUG:
            print(f"\n🔍 Retrieval:")
            print(f"   Expert: {routing['primary_expert']}")
            print(f"   Confiança: {routing['confidence']:.2f}")
        
        # Determinar collections para buscar
        collections_to_search = self.topic_router.route_to_collections(
            query,
            list(self.available_collections.keys())
        )
        
        if DEBUG:
            print(f"   Collections: {collections_to_search}")
        
        # Se não especificou collections, buscar em todas
        if not collections_to_search:
            collections_to_search = [
                (period, figure)
                for period, figures in self.available_collections.items()
                for figure in figures
            ]
        else:
            # Converter para tuplas (period, figure)
            tuples = []
            for col_spec in collections_to_search:
                if '/' in col_spec:
                    period, figure = col_spec.split('/')
                    tuples.append((period, figure))
                else:
                    # É só um período, buscar em todas as figuras
                    if col_spec in self.available_collections:
                        for figure in self.available_collections[col_spec]:
                            tuples.append((col_spec, figure))
            
            collections_to_search = tuples
        
        # Buscar em múltiplas collections
        results_by_collection = self.vectorstore_manager.search_in_multiple(
            collections_to_search,
            query,
            k_per_collection=TOP_K_DOCUMENTS
        )
        
        # Combinar resultados
        all_docs = []
        for collection_name, docs in results_by_collection.items():
            # Adicionar metadata de origem
            for doc in docs:
                doc.metadata['source_collection'] = collection_name
            all_docs.extend(docs)
        
        if DEBUG:
            print(f"   Total de documentos: {len(all_docs)}")
            for col, docs in results_by_collection.items():
                print(f"     • {col}: {len(docs)} docs")
        
        # Limitar ao TOP_K total
        return all_docs[:TOP_K_DOCUMENTS * 2]  # Permitir mais docs em multi-collection
    
    def _generate_answer(self, query: str, context: str, chat_history: str) -> str:
        """Gera resposta usando LLM"""
        prompt = RAG_PROMPT.format(
            context=context,
            chat_history=chat_history,
            question=query
        )
        
        if DEBUG:
            print(f"\n📝 Gerando resposta...")
        
        try:
            response = self.llm(prompt)
            answer = response.split("**Sua resposta:**")[-1].strip()
            return answer
        except Exception as e:
            return f"Desculpe, erro ao gerar resposta: {str(e)}"
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Processa uma query completa
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Dict com resposta e metadados
        """
        print(f"\n{'='*60}")
        print(f"💬 Query: {question}")
        print(f"{'='*60}")
        
        # Cumprimentos
        if self._is_greeting(question):
            import random
            answer = random.choice(GREETING_RESPONSES)
            return {
                "question": question,
                "answer": answer,
                "source_documents": [],
                "is_greeting": True
            }
        
        # Adicionar à memória
        self.memory.add_user_message(question)
        
        # Roteamento
        routing = self.topic_router.route_query(question)
        
        # Retrieval
        docs = self._retrieve_documents(question, routing)
        
        # Formatar contexto
        context = format_docs(docs)
        
        # Histórico
        chat_history = self.memory.get_formatted_history()
        
        # Gerar resposta
        answer = self._generate_answer(question, context, chat_history)
        
        # Adicionar resposta à memória
        self.memory.add_ai_message(answer)
        
        result = {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "routing": routing,
            "collections_used": list(set(
                doc.metadata.get('source_collection', 'unknown')
                for doc in docs
            ))
        }
        
        if DEBUG:
            print(f"\n✅ Resposta gerada!")
            print(f"   Collections usadas: {result['collections_used']}")
        
        return result
    
    def chat(self, message: str) -> str:
        """Interface simplificada - retorna apenas resposta"""
        result = self.query(message)
        return result["answer"]
    
    def clear_conversation(self):
        """Limpa histórico"""
        self.memory.clear_memory()
        print("🗑️  Conversa reiniciada!")
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do sistema"""
        vs_stats = self.vectorstore_manager.get_stats()
        router_stats = self.topic_router.get_routing_stats()
        memory_stats = self.memory.get_memory_stats()
        
        return {
            "model": MODEL_NAME,
            "device": DEVICE,
            "vectorstore": vs_stats,
            "router": router_stats,
            "memory": memory_stats,
        }


def main():
    """Teste standalone"""
    print("\n" + "="*60)
    print("🧪 TESTANDO MULTI-FIGURE RAG CHAIN")
    print("="*60)
    
    try:
        # Inicializar
        rag_chain = MultiFigureRAGChain()
        
        # Queries de teste
        test_queries = [
            "Quando Galileu nasceu?",
            "Quais são as três leis de Newton?",
            "O que é a teoria da relatividade de Einstein?",
            "Compare as contribuições de Galileu e Newton para a física",
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            result = rag_chain.query(query)
            print(f"\n💬 Pergunta: {result['question']}")
            print(f"🤖 Resposta: {result['answer'][:200]}...")
            print(f"📚 Collections usadas: {result['collections_used']}")
        
        # Estatísticas
        stats = rag_chain.get_stats()
        print(f"\n{'='*60}")
        print("📊 ESTATÍSTICAS")
        print(f"{'='*60}")
        print(f"Collections: {stats['vectorstore']['total_collections']}")
        print(f"Queries processadas: {stats['router']['total_queries']}")
        print(f"Expert mais usado: {stats['router']['most_used_expert']}")
        
        print("\n✅ Teste concluído!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
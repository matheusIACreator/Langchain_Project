"""
RAG Chain - Chain principal do sistema RAG
Integra vector store, LLM, memória e prompts para responder perguntas sobre Galileu
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Adiciona o diretório raiz ao path
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
from src.vectorstore import GalileuVectorStore
from src.memory.conversation_memory import GalileuConversationMemory
from src.prompts.rag_prompts import (
    RAG_PROMPT,
    CHAT_PROMPT,
    format_docs,
    GREETING_RESPONSES,
    OUT_OF_SCOPE_RESPONSE
)


class GalileuRAGChain:
    """
    Chain principal do sistema RAG sobre Galileu Galilei
    """
    
    def __init__(self, vectorstore: GalileuVectorStore = None, memory: GalileuConversationMemory = None):
        """
        Inicializa a RAG Chain
        
        Args:
            vectorstore: Vector store já inicializado (opcional)
            memory: Sistema de memória já inicializado (opcional)
        """
        print("\n" + "="*60)
        print("🚀 INICIALIZANDO RAG CHAIN - GALILEU GALILEI")
        print("="*60 + "\n")
        
        # Inicializar componentes
        self.vectorstore = vectorstore or self._load_vectorstore()
        self.memory = memory or GalileuConversationMemory(memory_type="window", k=5)
        self.llm = self._setup_llm()
        self.retriever = self._setup_retriever()
        
        print("\n✅ RAG Chain inicializada com sucesso!\n")
    
    def _load_vectorstore(self) -> GalileuVectorStore:
        """
        Carrega ou cria o vector store
        
        Returns:
            Vector store configurado
        """
        print("📂 Carregando vector store...")
        vs_manager = GalileuVectorStore()
        
        vectorstore = vs_manager.load_vectorstore()
        
        if vectorstore is None:
            raise ValueError(
                "❌ Vector store não encontrado!\n"
                "Execute 'python src/vectorstore.py' primeiro para criar o vector store."
            )
        
        return vs_manager
    
    def _setup_llm(self) -> HuggingFacePipeline:
        """
        Configura o modelo LLM (Llama)
        
        Returns:
            LLM configurado
        """
        print(f"🤖 Carregando modelo LLM: {MODEL_NAME}")
        print(f"   Device: {DEVICE}")
        print(f"   Isso pode levar alguns minutos na primeira vez...")
        
        try:
            # Carregar tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            
            # Configurar padding token se não existir
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Carregar modelo
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                trust_remote_code=True,
                **MODEL_KWARGS
            )
            
            # Criar pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False,
                **GENERATION_KWARGS
            )
            
            # Criar LangChain LLM
            llm = HuggingFacePipeline(pipeline=pipe)
            
            print(f"✅ Modelo carregado com sucesso!")
            
            return llm
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")
            raise
    
    def _setup_retriever(self):
        """
        Configura o retriever do vector store
        
        Returns:
            Retriever configurado
        """
        print(f"🔍 Configurando retriever (Top K: {TOP_K_DOCUMENTS})")
        
        retriever = self.vectorstore.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_DOCUMENTS}
        )
        
        print("✅ Retriever configurado!")
        
        return retriever
    
    def _is_greeting(self, query: str) -> bool:
        """
        Detecta se a query é um cumprimento
        
        Args:
            query: Texto da query
            
        Returns:
            True se for cumprimento
        """
        greetings = ["oi", "olá", "ola", "hey", "opa", "bom dia", "boa tarde", "boa noite", "e aí"]
        query_lower = query.lower().strip()
        
        return any(greeting in query_lower for greeting in greetings) and len(query_lower.split()) <= 3
    
    def _is_out_of_scope(self, query: str) -> bool:
        """
        Detecta se a query está fora do escopo (não é sobre Galileu)
        
        Args:
            query: Texto da query
            
        Returns:
            True se estiver fora do escopo
        """
        galileu_keywords = [
            "galileu", "galilei", "pisa", "telescópio", "telescopio",
            "júpiter", "jupiter", "lua", "sol", "astronomia",
            "física", "fisica", "igreja", "inquisição", "inquisicao"
        ]
        
        query_lower = query.lower()
        
        # Se menciona Galileu ou tópicos relacionados, está no escopo
        if any(keyword in query_lower for keyword in galileu_keywords):
            return False
        
        # Queries muito curtas podem ser cumprimentos
        if len(query_lower.split()) <= 3:
            return False
        
        # Caso contrário, pode estar fora do escopo
        return True
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Recupera documentos relevantes do vector store
        
        Args:
            query: Query de busca
            
        Returns:
            Lista de documentos relevantes
        """
        if DEBUG:
            print(f"\n🔍 Buscando documentos para: '{query}'")
        
        docs = self.retriever.get_relevant_documents(query)
        
        if DEBUG:
            print(f"✅ {len(docs)} documentos recuperados")
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Documento {i} ---")
                print(f"Página: {doc.metadata.get('page', 'N/A')}")
                print(f"Preview: {doc.page_content[:150]}...")
        
        return docs
    
    def generate_answer(self, query: str, context: str, chat_history: str) -> str:
        """
        Gera resposta usando o LLM
        
        Args:
            query: Pergunta do usuário
            context: Contexto recuperado
            chat_history: Histórico formatado
            
        Returns:
            Resposta gerada
        """
        # Montar prompt
        prompt = RAG_PROMPT.format(
            context=context,
            chat_history=chat_history,
            question=query
        )
        
        if DEBUG:
            print(f"\n📝 Prompt gerado ({len(prompt)} chars)")
            print(f"Preview: {prompt[:200]}...")
        
        # Gerar resposta
        try:
            response = self.llm(prompt)
            
            # Extrair apenas a resposta (remover o prompt)
            answer = response.split("**Sua resposta:**")[-1].strip()
            
            return answer
            
        except Exception as e:
            error_msg = f"Desculpe, ocorreu um erro ao gerar a resposta: {str(e)}"
            print(f"❌ Erro na geração: {error_msg}")
            return error_msg
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Processa uma query completa (retrieval + generation)
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Dict com resposta e metadados
        """
        print(f"\n{'='*60}")
        print(f"💬 PROCESSANDO QUERY")
        print(f"{'='*60}")
        print(f"Pergunta: {question}")
        
        # Verificar cumprimentos
        if self._is_greeting(question):
            import random
            answer = random.choice(GREETING_RESPONSES)
            
            # Não adiciona à memória cumprimentos simples
            
            return {
                "question": question,
                "answer": answer,
                "source_documents": [],
                "is_greeting": True
            }
        
        # Verificar se está fora do escopo
        if self._is_out_of_scope(question):
            answer = OUT_OF_SCOPE_RESPONSE
            
            return {
                "question": question,
                "answer": answer,
                "source_documents": [],
                "is_out_of_scope": True
            }
        
        # Adicionar pergunta à memória
        self.memory.add_user_message(question)
        
        # Recuperar documentos
        docs = self.retrieve_documents(question)
        
        # Formatar contexto
        context = format_docs(docs)
        
        # Obter histórico
        chat_history = self.memory.get_formatted_history()
        
        # Gerar resposta
        answer = self.generate_answer(question, context, chat_history)
        
        # Adicionar resposta à memória
        self.memory.add_ai_message(answer)
        
        # Retornar resultado completo
        result = {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "chat_history": chat_history,
        }
        
        print(f"\n✅ Query processada com sucesso!")
        print(f"Resposta: {answer[:100]}...")
        
        return result
    
    def chat(self, message: str) -> str:
        """
        Interface simplificada para chat (retorna apenas a resposta)
        
        Args:
            message: Mensagem do usuário
            
        Returns:
            Resposta do assistente
        """
        result = self.query(message)
        return result["answer"]
    
    def clear_conversation(self) -> None:
        """
        Limpa o histórico de conversação
        """
        self.memory.clear_memory()
        print("🗑️  Conversa reiniciada!")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do sistema
        
        Returns:
            Dict com estatísticas
        """
        memory_stats = self.memory.get_memory_stats()
        
        stats = {
            "model": MODEL_NAME,
            "device": DEVICE,
            "top_k_documents": TOP_K_DOCUMENTS,
            "memory": memory_stats,
        }
        
        return stats


def main():
    """
    Função principal para teste standalone
    """
    print("\n" + "="*60)
    print("🧪 TESTANDO RAG CHAIN")
    print("="*60 + "\n")
    
    try:
        # Inicializar RAG Chain
        rag_chain = GalileuRAGChain()
        
        # Teste de queries
        test_queries = [
            "Oi!",
            "Quando Galileu nasceu?",
            "Quais foram suas descobertas com o telescópio?",
            "O que aconteceu com a Igreja?",
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            result = rag_chain.query(query)
            print(f"\n💬 Pergunta: {result['question']}")
            print(f"🤖 Resposta: {result['answer']}\n")
            
            if result.get('source_documents'):
                print(f"📚 Documentos usados: {len(result['source_documents'])}")
        
        # Mostrar estatísticas
        print(f"\n{'='*60}")
        print("📊 ESTATÍSTICAS DO SISTEMA")
        print(f"{'='*60}")
        stats = rag_chain.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n✅ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

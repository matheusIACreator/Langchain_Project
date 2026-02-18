"""
Conversation Memory - Sistema de memória conversacional
Gerencia o histórico de mensagens do chat para manter contexto
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config.settings import MAX_MEMORY_MESSAGES, DEBUG


class GalileuConversationMemory:
    """
    Gerenciador de memória conversacional para o chatbot do Galileu
    """
    
    def __init__(self, memory_type: str = "window", k: int = None):
        """
        Inicializa o sistema de memória
        
        Args:
            memory_type: Tipo de memória ("buffer" ou "window")
                - buffer: Mantém todo o histórico
                - window: Mantém apenas as últimas k mensagens
            k: Número de mensagens a manter (apenas para window)
        """
        self.memory_type = memory_type
        self.k = k or MAX_MEMORY_MESSAGES
        
        print(f"🧠 Inicializando memória conversacional...")
        print(f"   Tipo: {memory_type}")
        if memory_type == "window":
            print(f"   Janela: {self.k} mensagens")
        
        # Inicializar histórico de chat
        self.chat_history = ChatMessageHistory()
        
        # Contador de interações
        self.interaction_count = 0
    
    def add_user_message(self, message: str) -> None:
        self.chat_history.add_user_message(message)
        self.interaction_count += 1
        self._enforce_window()  # ← aqui
        if DEBUG:
            print(f"\n💬 Usuário ({self.interaction_count}): {message[:50]}...")

    def add_ai_message(self, message: str) -> None:
        self.chat_history.add_ai_message(message)
        self._enforce_window()  # ← e aqui também
        if DEBUG:
            print(f"🤖 IA: {message[:50]}...")

    def _enforce_window(self):
        """Garante que a janela não ultrapasse k*2 mensagens."""
        if self.memory_type == "window":
            messages = self.chat_history.messages
            if len(messages) > self.k * 2:
                self.chat_history.messages = messages[-(self.k * 2):]
                
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Retorna as variáveis de memória para uso em chains
        
        Returns:
            Dict com histórico de mensagens
        """
        return {"chat_history": self.chat_history.messages}
    
    def get_chat_history(self) -> List[BaseMessage]:
        """
        Retorna o histórico de mensagens
        
        Returns:
            Lista de mensagens (HumanMessage e AIMessage)
        """
        return self.chat_history.messages
    
    def get_formatted_history(self) -> str:
        """
        Retorna o histórico formatado como string
        
        Returns:
            String formatada com o histórico
        """
        history = self.get_chat_history()
        
        if not history:
            return "Nenhuma conversa anterior."
        
        formatted = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Usuário: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistente: {msg.content}")
        
        return "\n".join(formatted)
    
    def clear_memory(self) -> None:
        """
        Limpa toda a memória conversacional
        """
        self.chat_history.clear()
        self.interaction_count = 0
        
        print("🗑️  Memória limpa!")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre a memória
        
        Returns:
            Dict com estatísticas
        """
        history = self.get_chat_history()
        
        stats = {
            "total_messages": len(history),
            "user_messages": sum(1 for msg in history if isinstance(msg, HumanMessage)),
            "ai_messages": sum(1 for msg in history if isinstance(msg, AIMessage)),
            "interactions": self.interaction_count,
            "memory_type": self.memory_type,
        }
        
        if self.memory_type == "window":
            stats["window_size"] = self.k
            stats["is_full"] = len(history) >= self.k * 2  # k * 2 pois cada interação = 2 msgs
        
        return stats
    
    def print_memory_stats(self) -> None:
        """
        Imprime estatísticas da memória
        """
        stats = self.get_memory_stats()
        
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS DA MEMÓRIA")
        print("="*60)
        print(f"Tipo de memória: {stats['memory_type']}")
        print(f"Total de mensagens: {stats['total_messages']}")
        print(f"  - Usuário: {stats['user_messages']}")
        print(f"  - IA: {stats['ai_messages']}")
        print(f"Interações: {stats['interactions']}")
        
        if self.memory_type == "window":
            print(f"Tamanho da janela: {stats['window_size']}")
            print(f"Janela cheia: {'Sim' if stats.get('is_full') else 'Não'}")
        
        print("="*60 + "\n")
    
    def print_chat_history(self) -> None:
        """
        Imprime o histórico de chat formatado
        """
        print("\n" + "="*60)
        print("💬 HISTÓRICO DO CHAT")
        print("="*60)
        print(self.get_formatted_history())
        print("="*60 + "\n")
    
    def save_to_dict(self) -> Dict[str, Any]:
        """
        Salva o estado da memória em um dicionário
        
        Returns:
            Dict com o estado da memória
        """
        history = self.get_chat_history()
        
        return {
            "memory_type": self.memory_type,
            "k": self.k,
            "interaction_count": self.interaction_count,
            "history": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content
                }
                for msg in history
            ]
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Carrega o estado da memória de um dicionário
        
        Args:
            data: Dict com o estado da memória
        """
        self.clear_memory()
        self.interaction_count = data.get("interaction_count", 0)
        
        for msg_data in data.get("history", []):
            if msg_data["type"] == "human":
                self.chat_history.add_user_message(msg_data["content"])
            else:
                self.chat_history.add_ai_message(msg_data["content"])
        
        print(f"✅ Memória carregada: {len(data.get('history', []))} mensagens")


def main():
    """
    Função principal para teste standalone
    """
    print("\n" + "="*60)
    print("🧪 TESTANDO SISTEMA DE MEMÓRIA")
    print("="*60 + "\n")
    
    # Criar memória com janela de 3 interações (6 mensagens)
    memory = GalileuConversationMemory(memory_type="window", k=3)
    
    # Simular conversação
    conversations = [
        ("Quando Galileu nasceu?", "Galileu Galilei nasceu em 15 de fevereiro de 1564, em Pisa, Itália."),
        ("Quais foram suas descobertas com o telescópio?", "Galileu descobriu as luas de Júpiter, as fases de Vênus, montanhas na Lua e manchas solares."),
        ("O que aconteceu com a Igreja?", "Galileu foi julgado pela Inquisição em 1633 por defender o heliocentrismo."),
        ("Quando ele morreu?", "Galileu faleceu em 8 de janeiro de 1642, aos 77 anos."),
    ]
    
    for user_msg, ai_msg in conversations:
        print(f"\n{'─'*60}")
        memory.add_user_message(user_msg)
        memory.add_ai_message(ai_msg)
    
    # Mostrar estatísticas
    memory.print_memory_stats()
    
    # Mostrar histórico
    memory.print_chat_history()
    
    # Testar save/load
    print("\n💾 Testando save/load...")
    saved_state = memory.save_to_dict()
    
    new_memory = GalileuConversationMemory(memory_type="window", k=3)
    new_memory.load_from_dict(saved_state)
    
    print("\n📋 Memória carregada - Verificando histórico:")
    new_memory.print_chat_history()
    
    print("\n✅ Teste concluído com sucesso!")


if __name__ == "__main__":
    main()
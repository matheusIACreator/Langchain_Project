"""
Main - Interface Gradio para o Chatbot Galileu Galilei
Interface web interativa para conversar sobre o pai da ciência moderna
"""

import sys
import os
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent))

import gradio as gr
from typing import List, Tuple

from src.chains.rag_chain import GalileuRAGChain
from config.settings import DEBUG


# ===== CONFIGURAÇÃO GLOBAL =====
# Inicializar o RAG Chain (será feito na primeira execução)
rag_chain = None


def initialize_rag_chain():
    """
    Inicializa o RAG Chain (lazy loading)
    """
    global rag_chain
    
    if rag_chain is None:
        print("\n🚀 Inicializando sistema RAG pela primeira vez...")
        try:
            rag_chain = GalileuRAGChain()
            print("✅ Sistema pronto!")
            return True
        except Exception as e:
            print(f"❌ Erro ao inicializar: {str(e)}")
            return False
    return True


# ===== FUNÇÕES DO CHAT =====

def chat_response(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Processa mensagem do usuário e retorna resposta
    
    Args:
        message: Mensagem do usuário
        history: Histórico do chat (não usado, mantido pela memória interna)
        
    Returns:
        Resposta do assistente
    """
    if not message or message.strip() == "":
        return "Por favor, faça uma pergunta sobre Galileu Galilei!"
    
    # Inicializar RAG Chain se necessário
    if not initialize_rag_chain():
        return "❌ Erro: Sistema não está inicializado. Verifique se o vector store foi criado."
    
    try:
        # Processar query
        response = rag_chain.chat(message)
        return response
        
    except Exception as e:
        error_msg = f"❌ Erro ao processar sua pergunta: {str(e)}"
        print(error_msg)
        if DEBUG:
            import traceback
            traceback.print_exc()
        return error_msg


def clear_conversation():
    """
    Limpa o histórico de conversação
    """
    if rag_chain is not None:
        rag_chain.clear_conversation()
        return "🗑️ Conversa reiniciada! Como posso ajudá-lo?"
    return "Sistema ainda não inicializado."


def get_system_stats() -> str:
    """
    Retorna estatísticas do sistema
    
    Returns:
        String formatada com estatísticas
    """
    if rag_chain is None:
        return "Sistema ainda não foi inicializado."
    
    try:
        stats = rag_chain.get_stats()
        memory_stats = stats.get("memory", {})
        
        stats_text = f"""
📊 **Estatísticas do Sistema**

**Modelo:** {stats.get('model', 'N/A')}
**Device:** {stats.get('device', 'N/A')}
**Top K Documents:** {stats.get('top_k_documents', 'N/A')}

**Memória:**
- Total de mensagens: {memory_stats.get('total_messages', 0)}
- Interações: {memory_stats.get('interactions', 0)}
- Tipo: {memory_stats.get('memory_type', 'N/A')}
"""
        
        if memory_stats.get('memory_type') == 'window':
            stats_text += f"- Tamanho da janela: {memory_stats.get('window_size', 'N/A')}\n"
        
        return stats_text
        
    except Exception as e:
        return f"Erro ao obter estatísticas: {str(e)}"


# ===== EXEMPLOS DE PERGUNTAS =====

EXAMPLE_QUESTIONS = [
    "Quando e onde Galileu Galilei nasceu?",
    "Quais foram as principais descobertas de Galileu com o telescópio?",
    "O que aconteceu entre Galileu e a Igreja Católica?",
    "Quais invenções Galileu criou?",
    "Como Galileu contribuiu para a física?",
    "Quando e como Galileu morreu?",
    "Qual foi o papel de Galileu na revolução científica?",
    "O que é o método científico de Galileu?",
]


# ===== INTERFACE GRADIO =====

def create_interface():
    """
    Cria a interface Gradio
    
    Returns:
        Interface Gradio configurada
    """
    
    # CSS customizado para melhorar a aparência
    custom_css = """
    .container {
        max-width: 900px;
        margin: auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .examples {
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        color: #666;
        font-size: 0.9em;
    }
    """
    
    # Criar interface
    with gr.Blocks(title="Galileu Galilei Chatbot") as demo:
        
        # Header
        gr.Markdown("""
        <div class="header">
            <h1>🔭 Galileu Galilei Chatbot</h1>
            <p>Converse com um assistente especializado no pai da ciência moderna</p>
        </div>
        """)
        
        # Informações sobre o sistema
        with gr.Accordion("ℹ️ Sobre este chatbot", open=False):
            gr.Markdown("""
            Este chatbot utiliza **Retrieval-Augmented Generation (RAG)** para responder perguntas sobre Galileu Galilei.
            
            **Tecnologias utilizadas:**
            - 🤖 **LLM:** Meta Llama-3.1-8B-Instruct
            - 🗄️ **Vector Store:** ChromaDB
            - 🔗 **Framework:** LangChain
            - 🧠 **Memória:** Conversacional com histórico
            - 📄 **Fonte:** Documento detalhado sobre a vida e obra de Galileu
            
            **Como usar:**
            1. Digite sua pergunta sobre Galileu Galilei
            2. O sistema busca informações relevantes no documento
            3. O LLM gera uma resposta contextualizada
            4. Você pode fazer perguntas de acompanhamento - o chatbot mantém o contexto!
            """)
        
        # Interface de chat principal
        chatbot = gr.ChatInterface(
            fn=chat_response,
            examples=EXAMPLE_QUESTIONS,
            title="🔭 Galileu Galilei Chatbot",
            description="Pergunte sobre a vida, descobertas e legado de Galileu Galilei",
        )
        
        # Seção de estatísticas
        with gr.Accordion("📊 Estatísticas do Sistema", open=False):
            stats_display = gr.Markdown("Clique em 'Atualizar Estatísticas' para ver informações do sistema.")
            stats_btn = gr.Button("🔄 Atualizar Estatísticas")
            stats_btn.click(fn=get_system_stats, inputs=None, outputs=stats_display)
        
        # Footer com informações
        gr.Markdown("""
        <div class="footer">
            <p><strong>Desenvolvido por:</strong> Matheus Masago</p>
            <p>📚 Projeto educacional de RAG System com LangChain</p>
            <p>💡 <em>Dica:</em> Faça perguntas específicas sobre a vida, descobertas e legado de Galileu!</p>
        </div>
        """)
    
    return demo


# ===== FUNÇÃO PRINCIPAL =====

def main():
    """
    Função principal - inicializa e lança a interface
    """
    print("\n" + "="*60)
    print("🚀 INICIANDO CHATBOT GALILEU GALILEI")
    print("="*60 + "\n")
    
    # Verificar se o vector store existe
    vectorstore_path = Path("data/vectorstore")
    if not vectorstore_path.exists() or not any(vectorstore_path.iterdir()):
        print("⚠️  ATENÇÃO: Vector store não encontrado!")
        print("\n📋 Execute os seguintes comandos primeiro:")
        print("   1. python src/document_loader.py")
        print("   2. python src/vectorstore.py")
        print("\nDepois execute este script novamente.")
        return
    
    # Criar interface
    demo = create_interface()
    
    # Informações de lançamento
    print("\n📍 Informações:")
    print("   - Interface: Gradio")
    print("   - Modelo: Llama-3.1-8B-Instruct")
    print("   - Vector Store: ChromaDB")
    print("   - Memória: Conversacional")
    
    print("\n🌐 Abrindo interface web...")
    print("   Acesse pelo navegador quando estiver pronto!")
    print("\n💡 Dica: Use Ctrl+C para encerrar\n")
    
    # Lançar interface
    # share=True cria um link público temporário (útil para demonstrações)
    # share=False mantém apenas local
    demo.launch(
        server_name="0.0.0.0",  # Permite acesso de outros dispositivos na rede
        server_port=7860,
        share=False,  # Mude para True se quiser link público
        show_error=True,
        quiet=False,
        inbrowser=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Encerrando aplicação...")
        print("Até logo!")
    except Exception as e:
        print(f"\n❌ Erro fatal: {str(e)}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)
"""
Main - Interface Gradio para o Chatbot Galileu Galilei COM FEEDBACK
Interface web interativa com sistema de coleta de feedback para RLHF
VERSÃO COMPATÍVEL - Funciona com Gradio 3.x e 4.x
"""

import sys
import os
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent))

import gradio as gr
from typing import List, Tuple, Optional
import uuid

from src.chains.rag_chain import GalileuRAGChain
from src.feedback.feedback_collector import FeedbackCollector
from config.settings import DEBUG


# ===== CONFIGURAÇÃO GLOBAL =====
rag_chain = None
feedback_collector = None
current_session_id = None
last_query = None
last_response = None


def initialize_system():
    """
    Inicializa o RAG Chain e Feedback Collector
    """
    global rag_chain, feedback_collector, current_session_id
    
    if rag_chain is None:
        print("\n🚀 Inicializando sistema RAG pela primeira vez...")
        try:
            rag_chain = GalileuRAGChain()
            feedback_collector = FeedbackCollector()
            current_session_id = str(uuid.uuid4())
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
    """
    global last_query, last_response
    
    if not message or message.strip() == "":
        return "Por favor, faça uma pergunta sobre Galileu Galilei!"
    
    if not initialize_system():
        return "❌ Erro: Sistema não está inicializado. Verifique se o vector store foi criado."
    
    try:
        # Processar query
        response = rag_chain.chat(message)
        
        # Armazenar para feedback posterior
        last_query = message
        last_response = response
        
        return response
        
    except Exception as e:
        error_msg = f"❌ Erro ao processar sua pergunta: {str(e)}"
        print(error_msg)
        if DEBUG:
            import traceback
            traceback.print_exc()
        return error_msg


def submit_thumbs_feedback(is_thumbs_up: bool) -> str:
    """
    Submete feedback de thumbs up/down
    """
    global last_query, last_response, current_session_id
    
    if not last_query or not last_response:
        return "❌ Nenhuma resposta anterior para avaliar."
    
    try:
        feedback_collector.add_feedback(
            query=last_query,
            response=last_response,
            thumbs_up=is_thumbs_up,
            session_id=current_session_id
        )
        
        emoji = "👍" if is_thumbs_up else "👎"
        return f"{emoji} Obrigado pelo feedback!"
        
    except Exception as e:
        return f"❌ Erro ao registrar feedback: {str(e)}"


def submit_rating_feedback(rating: int, comment: str = "") -> str:
    """
    Submete feedback com rating e comentário
    """
    global last_query, last_response, current_session_id
    
    if not last_query or not last_response:
        return "❌ Nenhuma resposta anterior para avaliar."
    
    if rating < 1 or rating > 5:
        return "❌ Rating deve estar entre 1 e 5."
    
    try:
        feedback_collector.add_feedback(
            query=last_query,
            response=last_response,
            rating=rating,
            comment=comment if comment else None,
            session_id=current_session_id
        )
        
        return f"⭐ Obrigado! Rating {rating}/5 registrado."
        
    except Exception as e:
        return f"❌ Erro ao registrar feedback: {str(e)}"


def clear_conversation():
    """
    Limpa o histórico de conversação
    """
    global current_session_id
    
    if rag_chain is not None:
        rag_chain.clear_conversation()
        current_session_id = str(uuid.uuid4())  # Nova sessão
        return "🗑️ Conversa reiniciada! Como posso ajudá-lo?"
    return "Sistema ainda não inicializado."


def get_system_stats() -> str:
    """
    Retorna estatísticas do sistema
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


def get_feedback_stats() -> str:
    """
    Retorna estatísticas de feedback coletado
    """
    if feedback_collector is None:
        return "Sistema de feedback ainda não foi inicializado."
    
    try:
        stats = feedback_collector.get_feedback_stats()
        
        stats_text = f"""
📊 **Estatísticas de Feedback**

**Total de feedbacks:** {stats['total_feedbacks']}
**Com rating:** {stats['with_rating']}
**Rating médio:** {stats['avg_rating']}/5.0
**👍 Thumbs up:** {stats['thumbs_up']}
**👎 Thumbs down:** {stats['thumbs_down']}
**Com comentário:** {stats['with_comment']}
**Pares de preferência:** {stats['preference_pairs']}
"""
        
        return stats_text
        
    except Exception as e:
        return f"Erro ao obter estatísticas de feedback: {str(e)}"


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
    Cria a interface Gradio (compatível com versões antigas)
    """
    
    with gr.Blocks(title="Galileu Galilei Chatbot + Feedback") as demo:
        
        # Header
        gr.Markdown("""
        # 🔭 Galileu Galilei Chatbot
        
        **Converse com um assistente especializado no pai da ciência moderna**
        
        ✨ *Agora com sistema de feedback para melhoria contínua!*
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
            - 📊 **Feedback:** Sistema de coleta para RLHF futuro
            - 📄 **Fonte:** Documento detalhado sobre a vida e obra de Galileu
            
            **Como usar:**
            1. Digite sua pergunta sobre Galileu Galilei
            2. O sistema busca informações relevantes no documento
            3. O LLM gera uma resposta contextualizada
            4. **NOVO:** Avalie a resposta com thumbs up/down ou rating!
            """)
        
        # Interface de chat principal
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                placeholder="Pergunte sobre Galileu Galilei...",
                label="Sua pergunta"
            )
            
            with gr.Row():
                submit = gr.Button("🚀 Enviar", variant="primary")
                clear = gr.Button("🗑️ Limpar conversa")
            
            # Sistema de feedback
            gr.Markdown("---")
            gr.Markdown("### 📊 Avalie a última resposta")
            
            with gr.Row():
                thumbs_up_btn = gr.Button("👍 Boa resposta", variant="primary")
                thumbs_down_btn = gr.Button("👎 Resposta ruim")
            
            feedback_status = gr.Textbox(label="Status do feedback", interactive=False)
            
            with gr.Accordion("✍️ Feedback detalhado (opcional)", open=False):
                with gr.Row():
                    rating_slider = gr.Slider(
                        minimum=1, 
                        maximum=5, 
                        step=1, 
                        value=3,
                        label="Rating (1-5 estrelas)"
                    )
                comment_box = gr.Textbox(
                    label="Comentário (opcional)",
                    placeholder="O que você achou da resposta? Como poderia melhorar?",
                    lines=3
                )
                rating_submit = gr.Button("⭐ Enviar feedback detalhado")
            
            # Exemplos
            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=msg,
                label="💡 Exemplos de perguntas"
            )
        
        # Tab de estatísticas
        with gr.Tab("📊 Estatísticas"):
            gr.Markdown("### 📈 Estatísticas do Sistema")
            
            with gr.Row():
                with gr.Column():
                    stats_display = gr.Markdown("Clique em 'Atualizar' para ver estatísticas.")
                    stats_btn = gr.Button("🔄 Atualizar Estatísticas do Sistema")
                
                with gr.Column():
                    feedback_stats_display = gr.Markdown("Clique em 'Atualizar' para ver feedback.")
                    feedback_stats_btn = gr.Button("📊 Atualizar Estatísticas de Feedback")
            
            gr.Markdown("---")
            gr.Markdown("""
            ### 🎯 Sobre o Sistema de Feedback
            
            O feedback que você fornece é armazenado e pode ser usado para:
            - **Análise de qualidade:** Identificar pontos fortes e fracos
            - **RLHF (futuro):** Treinar modelos reward e melhorar o chatbot
            - **DPO (futuro):** Direct Preference Optimization
            - **Melhoria contínua:** Ajustar prompts e retrieval
            
            Seus feedbacks são anônimos e usados apenas para melhorar o sistema! 🙏
            """)
        
        # Conectar eventos
        def respond(message, chat_history):
            response = chat_response(message, chat_history)
            chat_history.append((message, response))
            return "", chat_history
        
        def clear_chat():
            clear_conversation()
            return None, "🗑️ Conversa reiniciada!"
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat, None, [chatbot, feedback_status])
        
        thumbs_up_btn.click(
            lambda: submit_thumbs_feedback(True),
            None,
            feedback_status
        )
        
        thumbs_down_btn.click(
            lambda: submit_thumbs_feedback(False),
            None,
            feedback_status
        )
        
        rating_submit.click(
            submit_rating_feedback,
            [rating_slider, comment_box],
            feedback_status
        )
        
        stats_btn.click(get_system_stats, None, stats_display)
        feedback_stats_btn.click(get_feedback_stats, None, feedback_stats_display)
        
        # Footer
        gr.Markdown("""
        ---
        
        **Desenvolvido por:** Matheus Masago  
        📚 Projeto educacional de RAG System com LangChain + RLHF  
        💡 *Seus feedbacks ajudam a melhorar o sistema!*
        """)
    
    return demo


# ===== FUNÇÃO PRINCIPAL =====

def main():
    """
    Função principal - inicializa e lança a interface
    """
    print("\n" + "="*60)
    print("🚀 INICIANDO CHATBOT GALILEU GALILEI + FEEDBACK")
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
    print("   - 🆕 Sistema de Feedback: Ativo")
    
    print("\n🌐 Abrindo interface web...")
    print("   Acesse pelo navegador quando estiver pronto!")
    print("\n💡 Dicas:")
    print("   - Use Ctrl+C para encerrar")
    print("   - Avalie as respostas para melhorar o sistema!")
    print("   - Seus feedbacks são armazenados em data/feedback/\n")
    
    # Lançar interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
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
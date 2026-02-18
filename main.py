"""
Main - Interface Gradio v2.0
Sistema RAG Multi-Figura: Galileu, Newton, Einstein
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import gradio as gr
from typing import List, Tuple

from src.chains.rag_chain_multi import MultiFigureRAGChain
from config.settings import DEBUG, MODEL_NAME, DEVICE

# ===== CONFIGURAÇÃO GLOBAL =====
rag_chain = None

# Diretório do vectorstore multi-collection
VECTORSTORE_BASE = Path("data/vectorstore")
EXPECTED_COLLECTIONS = [
    "renaissance/galileo_galilei",
    "enlightenment/isaac_newton",
    "modern_era/albert_einstein",
]

# Figuras e descrições para exibir na UI
FIGURES_INFO = {
    "galileo_galilei": {
        "label": "🔭 Galileu Galilei",
        "period": "Renascimento",
        "years": "1564–1642",
        "description": "Pai da ciência moderna, astrônomo e físico italiano.",
    },
    "isaac_newton": {
        "label": "🍎 Isaac Newton",
        "period": "Iluminismo",
        "years": "1643–1727",
        "description": "Formulou as leis do movimento e da gravitação universal.",
    },
    "albert_einstein": {
        "label": "⚛️ Albert Einstein",
        "period": "Era Moderna",
        "years": "1879–1955",
        "description": "Autor da teoria da relatividade e pioneiro da física quântica.",
    },
}

# Exemplos de perguntas organizados por tipo
EXAMPLE_QUESTIONS = [
    # Figura única
    "Quando e onde Galileu Galilei nasceu?",
    "Quais foram as principais descobertas de Newton?",
    "O que Einstein contribuiu para a física quântica?",
    # Comparativas
    "Compare as contribuições de Newton e Einstein para a física.",
    "Qual a diferença entre a visão de gravidade de Newton e Einstein?",
    "Como a física evoluiu do Renascimento até a Era Moderna?",
    # Contextuais
    "O que aconteceu entre Galileu e a Igreja Católica?",
    "Como Newton desenvolveu o cálculo?",
    "Por que Einstein ganhou o Nobel de Física?",
]


# ===== INICIALIZAÇÃO =====

def check_vectorstore() -> Tuple[bool, str]:
    """
    Verifica se o vectorstore multi-collection está disponível.

    Returns:
        (ok: bool, mensagem: str)
    """
    if not VECTORSTORE_BASE.exists():
        return False, (
            f"Diretório `{VECTORSTORE_BASE}` não encontrado.\n\n"
            "Execute:\n"
            "```\n"
            "python src/ingestion/pipeline.py\n"
            "python src/vectorstore.py --mode multi\n"
            "```"
        )

    found = []
    missing = []
    for col in EXPECTED_COLLECTIONS:
        col_path = VECTORSTORE_BASE / col
        if col_path.exists() and any(col_path.iterdir()):
            found.append(col)
        else:
            missing.append(col)

    if not found:
        return False, (
            "Nenhuma collection encontrada em `data/vectorstore/`.\n\n"
            "Execute:\n"
            "```\n"
            "python src/ingestion/pipeline.py\n"
            "python src/vectorstore.py --mode multi\n"
            "```"
        )

    if missing:
        msg = f"Collections disponíveis: {len(found)}/{len(EXPECTED_COLLECTIONS)}\n"
        msg += f"Faltando: {', '.join(missing)}"
        # Parcialmente OK — o sistema consegue rodar com menos figuras
        return True, msg

    return True, f"Todas as {len(found)} collections prontas."


def initialize_rag_chain() -> Tuple[bool, str]:
    """
    Inicializa o MultiFigureRAGChain (lazy loading).

    Returns:
        (sucesso: bool, mensagem: str)
    """
    global rag_chain

    if rag_chain is not None:
        return True, "Sistema já inicializado."

    print("\n🚀 Inicializando Multi-Figure RAG Chain...")
    try:
        rag_chain = MultiFigureRAGChain()
        print("✅ Sistema pronto!")
        return True, "Sistema inicializado com sucesso!"
    except Exception as e:
        msg = f"Erro ao inicializar: {str(e)}"
        print(f"❌ {msg}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return False, msg


# ===== FUNÇÕES DO CHAT =====

def chat_response(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Processa mensagem e retorna resposta do sistema multi-figura.
    """
    if not message or not message.strip():
        return "Por favor, faça uma pergunta sobre Galileu, Newton ou Einstein!"

    # Inicializar chain se necessário
    ok, msg = initialize_rag_chain()
    if not ok:
        return f"❌ Sistema não inicializado.\n\n{msg}"

    try:
        response = rag_chain.chat(message)
        return response
    except Exception as e:
        error_msg = f"❌ Erro ao processar sua pergunta: {str(e)}"
        print(error_msg)
        if DEBUG:
            import traceback
            traceback.print_exc()
        return error_msg


def clear_conversation() -> str:
    """Limpa o histórico de conversação."""
    if rag_chain is not None:
        rag_chain.clear_conversation()
        return "🗑️ Conversa reiniciada! Como posso ajudá-lo?"
    return "Sistema ainda não inicializado."


def get_system_stats() -> str:
    """Retorna estatísticas detalhadas do sistema v2.0."""
    if rag_chain is None:
        return "⚠️ Sistema ainda não foi inicializado. Envie uma mensagem primeiro."

    try:
        stats = rag_chain.get_stats()
        memory_stats = stats.get("memory", {})
        vs_stats = stats.get("vectorstore", {})
        router_stats = stats.get("router", {})

        # Montar texto de stats
        lines = [
            "## 📊 Estatísticas do Sistema v2.0",
            "",
            f"**Modelo LLM:** `{stats.get('model', 'N/A')}`",
            f"**Device:** `{stats.get('device', 'N/A')}`",
            "",
            "### 🗄️ Vector Store",
            f"- Collections totais: **{vs_stats.get('total_collections', 'N/A')}**",
            f"- Collections carregadas: **{vs_stats.get('collections_loaded', 'N/A')}**",
            f"- Embedding model: `{vs_stats.get('embedding_model', 'N/A')}`",
        ]

        collections = vs_stats.get("collections_list", [])
        if collections:
            lines.append("")
            lines.append("**Collections disponíveis:**")
            for col in sorted(collections):
                lines.append(f"  - `{col}`")

        lines += [
            "",
            "### 🧭 Topic Router",
            f"- Queries roteadas: **{router_stats.get('total_queries', 0)}**",
            f"- Cache hits: **{router_stats.get('cache_hits', 0)}**",
            "",
            "### 🧠 Memória Conversacional",
            f"- Total de mensagens: **{memory_stats.get('total_messages', 0)}**",
            f"- Interações: **{memory_stats.get('interactions', 0)}**",
            f"- Tipo: `{memory_stats.get('memory_type', 'N/A')}`",
        ]

        if memory_stats.get("memory_type") == "window":
            lines.append(f"- Tamanho da janela: **{memory_stats.get('window_size', 'N/A')}**")

        return "\n".join(lines)

    except Exception as e:
        return f"Erro ao obter estatísticas: {str(e)}"


def get_routing_info(message: str) -> str:
    """
    Retorna informação de roteamento para uma query (modo debug).
    Útil para entender como o Topic Router classificou a pergunta.
    """
    if not message or not message.strip():
        return ""
    if rag_chain is None:
        return ""
    try:
        routing = rag_chain.topic_router.route_query(message)
        expert = routing.get("primary_expert", "N/A")
        confidence = routing.get("confidence", 0)
        reason = routing.get("routing_reason", "")
        secondary = routing.get("secondary_experts", [])

        lines = [f"🧭 **Expert:** `{expert}` (confiança: {confidence:.0%})"]
        if secondary:
            lines.append(f"🔀 **Secundários:** {', '.join(secondary)}")
        if reason:
            lines.append(f"💡 **Razão:** {reason}")
        return "\n".join(lines)
    except Exception:
        return ""


# ===== INTERFACE GRADIO =====

def create_interface() -> gr.Blocks:
    """Cria a interface Gradio v2.0."""

    custom_css = """
    .container { max-width: 960px; margin: auto; }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .figure-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        background: #fafafa;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 16px;
        color: #888;
        font-size: 0.85em;
    }
    """

    with gr.Blocks(title="Cientistas Históricos — RAG Multi-Figura", css=custom_css) as demo:

        # ── Header ──
        gr.Markdown("""
        <div class="header">
            <h1>🏛️ Cientistas Históricos</h1>
            <p>Sistema RAG Multi-Figura · Galileu · Newton · Einstein</p>
        </div>
        """)

        # ── Cards das figuras ──
        with gr.Row():
            for fig_key, info in FIGURES_INFO.items():
                with gr.Column():
                    gr.Markdown(f"""
**{info['label']}**
*{info['period']} · {info['years']}*

{info['description']}
                    """)

        gr.Markdown("---")

        # ── Sobre o sistema ──
        with gr.Accordion("ℹ️ Sobre este sistema", open=False):
            gr.Markdown(f"""
**Versão:** 2.0 · Multi-Figura

**Como funciona:**
1. Sua pergunta é analisada pelo **Topic Router**, que identifica quais figuras e experts são relevantes.
2. O **Hybrid Retriever** busca os trechos mais relevantes nas collections ChromaDB (busca semântica + BM25).
3. O **LLM** (`{MODEL_NAME}`) gera uma resposta contextualizada com base nos documentos recuperados.
4. A **memória conversacional** mantém o contexto ao longo da conversa.

**Você pode:**
- Perguntar sobre uma figura específica: *"Quem foi Newton?"*
- Fazer perguntas comparativas: *"Compare Newton e Einstein"*
- Explorar períodos: *"Como a física evoluiu do Renascimento à Era Moderna?"*
- Fazer perguntas de acompanhamento — o sistema mantém o contexto!
            """)

        # ── Chat principal ──
        chatbot = gr.ChatInterface(
            fn=chat_response,
            examples=EXAMPLE_QUESTIONS,
            title="",
            description="💬 Pergunte sobre Galileu Galilei, Isaac Newton ou Albert Einstein",
            retry_btn="🔄 Tentar novamente",
            undo_btn="↩️ Desfazer",
            clear_btn="🗑️ Limpar conversa",
        )

        gr.Markdown("---")

        # ── Painel inferior: Stats + Debug ──
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("📊 Estatísticas do Sistema", open=False):
                    stats_display = gr.Markdown("*Clique em 'Atualizar' para ver as estatísticas.*")
                    stats_btn = gr.Button("🔄 Atualizar Estatísticas", size="sm")
                    stats_btn.click(fn=get_system_stats, inputs=None, outputs=stats_display)

            with gr.Column(scale=1):
                with gr.Accordion("🧭 Debug: Topic Router", open=False):
                    gr.Markdown("*Veja como o sistema classificou sua última pergunta.*")
                    debug_input = gr.Textbox(
                        placeholder="Cole aqui sua pergunta para ver o roteamento...",
                        label="Pergunta",
                        lines=2,
                    )
                    debug_output = gr.Markdown()
                    debug_btn = gr.Button("🔍 Analisar Roteamento", size="sm")
                    debug_btn.click(fn=get_routing_info, inputs=debug_input, outputs=debug_output)

        # ── Footer ──
        gr.Markdown("""
        <div class="footer">
            <p><strong>Desenvolvido por Matheus Masago</strong> · RAG System v2.0 com LangChain</p>
            <p>ChromaDB · all-MiniLM-L6-v2 · BM25 · Reciprocal Rank Fusion</p>
        </div>
        """)

    return demo


# ===== FUNÇÃO PRINCIPAL =====

def main():
    print("\n" + "=" * 60)
    print("🚀 INICIANDO SISTEMA RAG MULTI-FIGURA v2.0")
    print("=" * 60 + "\n")

    # Verificar vectorstore antes de subir a interface
    vs_ok, vs_msg = check_vectorstore()
    if not vs_ok:
        print(f"❌ ERRO: Vectorstore não encontrado ou vazio.\n")
        print(vs_msg)
        print("\n💡 Execute os comandos acima e tente novamente.")
        return

    print(f"✅ Vectorstore: {vs_msg}")
    print(f"🤖 Modelo: {MODEL_NAME}")
    print(f"⚙️  Device: {DEVICE}")
    print("\n📌 A chain será inicializada na primeira mensagem (lazy loading).")
    print("\n🌐 Iniciando interface web...")
    print("💡 Use Ctrl+C para encerrar\n")

    demo = create_interface()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Encerrando aplicação... Até logo!")
    except Exception as e:
        print(f"\n❌ Erro fatal: {str(e)}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)
"""
api.py — FastAPI Backend para o sistema RAG Multi-Figura v2.0

Endpoints:
  POST /chat                     — Enviar mensagem e receber resposta
  DELETE /conversation/{session} — Limpar histórico de uma sessão
  GET  /figures                  — Listar figuras disponíveis
  GET  /stats                    — Estatísticas do sistema
  GET  /health                   — Health check

Execução:
  uvicorn api:app --reload --port 8000
"""

import sys
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chains.rag_chain_multi import MultiFigureRAGChain
from config.settings import DEBUG, MODEL_NAME, DEVICE


# ══════════════════════════════════════════════════════
# MODELOS PYDANTIC — Request / Response
# ══════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Mensagem do utilizador")
    session_id: Optional[str] = Field(None, description="ID de sessão (gerado automaticamente se omitido)")

    model_config = {"json_schema_extra": {"example": {"message": "Quando Newton nasceu?", "session_id": "abc-123"}}}


class RoutingInfo(BaseModel):
    primary_expert: str
    secondary_experts: List[str]
    confidence: float
    routing_reason: str


class SourceDocument(BaseModel):
    content: str
    source_collection: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    collections_used: List[str]
    retrieval_mode: str
    routing: RoutingInfo
    source_documents: List[SourceDocument]
    is_greeting: bool
    latency_ms: float


class ConversationDeleteResponse(BaseModel):
    session_id: str
    deleted: bool
    message: str


class FigureInfo(BaseModel):
    key: str
    name: str
    period: str
    collection: str
    years: str
    description: str


class FiguresResponse(BaseModel):
    figures: List[FigureInfo]
    total: int


class SystemStats(BaseModel):
    model: str
    device: str
    retrieval_mode: str
    vectorstore: Dict[str, Any]
    router: Dict[str, Any]
    memory: Dict[str, Any]
    hybrid_retriever: Dict[str, Any]
    active_sessions: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_ready: bool
    active_sessions: int
    uptime_seconds: float


# ══════════════════════════════════════════════════════
# ESTADO GLOBAL
# ══════════════════════════════════════════════════════

class AppState:
    """Estado partilhado da aplicação."""

    def __init__(self):
        self.chain: Optional[MultiFigureRAGChain] = None
        # Cada sessão tem a sua própria instância de memória
        # Por simplicidade, usamos a chain global e gerimos sessões via dicionário
        self.sessions: Dict[str, Dict] = {}
        self.start_time: float = time.time()

    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """Retorna session_id existente ou cria um novo."""
        if not session_id or session_id not in self.sessions:
            session_id = session_id or str(uuid.uuid4())
            self.sessions[session_id] = {
                "created_at": time.time(),
                "message_count": 0,
            }
        return session_id

    def increment_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] += 1

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time


state = AppState()


# ══════════════════════════════════════════════════════
# LIFESPAN — Carrega o modelo ao iniciar
# ══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa a chain ao arrancar e faz cleanup ao parar."""
    print("\n🚀 Iniciando API — carregando modelo...")
    try:
        state.chain = MultiFigureRAGChain()
        print("✅ Chain carregada! API pronta.")
    except Exception as e:
        print(f"❌ Erro ao carregar chain: {e}")
        raise

    yield  # API está a correr

    print("🛑 A encerrar API...")
    state.chain = None


# ══════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════

app = FastAPI(
    title="Cientistas Históricos — RAG API",
    description="Backend RAG Multi-Figura: Galileu · Newton · Einstein",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — permite qualquer origem em desenvolvimento
# Em produção, substituir ["*"] pela URL do frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _require_chain():
    """Lança 503 se a chain não estiver carregada."""
    if state.chain is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado. Tente novamente em instantes.")


def _format_source_documents(docs) -> List[SourceDocument]:
    """Converte documentos LangChain para o modelo Pydantic."""
    result = []
    for doc in docs:
        result.append(SourceDocument(
            content=doc.page_content[:500],  # truncar para não poluir a resposta
            source_collection=doc.metadata.get("source_collection", "unknown"),
            metadata={k: v for k, v in doc.metadata.items() if k != "source_collection"},
        ))
    return result


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Verifica se a API está operacional."""
    chain_ok = state.chain is not None
    vs_ok = chain_ok and len(state.chain.available_collections) > 0

    return HealthResponse(
        status="ok" if chain_ok else "loading",
        model_loaded=chain_ok,
        vectorstore_ready=vs_ok,
        active_sessions=len(state.sessions),
        uptime_seconds=round(state.uptime, 2),
    )


@app.get("/figures", response_model=FiguresResponse, tags=["Figuras"])
async def list_figures():
    """Lista todas as figuras históricas disponíveis no sistema."""
    _require_chain()

    # Metadados estáticos das figuras
    FIGURES_METADATA = {
        "galileo_galilei": {
            "name": "Galileu Galilei",
            "period": "Renascimento",
            "collection": "renaissance/galileo_galilei",
            "years": "1564–1642",
            "description": "Pai da ciência moderna. Astrônomo, físico e matemático italiano.",
        },
        "isaac_newton": {
            "name": "Isaac Newton",
            "period": "Iluminismo",
            "collection": "enlightenment/isaac_newton",
            "years": "1643–1727",
            "description": "Formulou as leis do movimento e da gravitação universal.",
        },
        "albert_einstein": {
            "name": "Albert Einstein",
            "period": "Era Moderna",
            "collection": "modern_era/albert_einstein",
            "years": "1879–1955",
            "description": "Autor da teoria da relatividade. Prémio Nobel de Física em 1921.",
        },
    }

    available = state.chain.available_collections
    figures = []

    for key, meta in FIGURES_METADATA.items():
        period = meta["collection"].split("/")[0]
        # Só incluir figuras com collection disponível
        if any(key in col for cols in available.values() for col in cols):
            figures.append(FigureInfo(key=key, **meta))

    return FiguresResponse(figures=figures, total=len(figures))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    _require_chain()

    session_id = state.get_or_create_session(request.session_id)
    start = time.time()

    try:
        result = state.chain.query(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar query: {str(e)}")

    latency = (time.time() - start) * 1000
    state.increment_session(session_id)

    # Cumprimentos não têm routing nem collections
    is_greeting = result.get("is_greeting", False)
    routing_raw = result.get("routing", {})

    routing = RoutingInfo(
        primary_expert=routing_raw.get("primary_expert", "greeting"),
        secondary_experts=routing_raw.get("secondary_experts", []),
        confidence=routing_raw.get("confidence", 1.0),
        routing_reason=routing_raw.get("routing_reason", "Cumprimento detectado"),
    )

    return ChatResponse(
        answer=result["answer"],
        session_id=session_id,
        collections_used=result.get("collections_used", []),
        retrieval_mode=result.get("retrieval_mode", "none"),
        routing=routing,
        source_documents=_format_source_documents(result.get("source_documents", [])),
        is_greeting=is_greeting,
        latency_ms=round(latency, 2),
    )

@app.delete("/conversation/{session_id}", response_model=ConversationDeleteResponse, tags=["Chat"])
async def clear_conversation(session_id: str):
    """
    Limpa o histórico de memória de uma sessão.
    Útil para reiniciar o contexto sem criar uma nova sessão.
    """
    _require_chain()

    deleted = state.delete_session(session_id)

    if deleted:
        # Limpar também a memória interna da chain
        state.chain.clear_conversation()

    return ConversationDeleteResponse(
        session_id=session_id,
        deleted=deleted,
        message="Conversa reiniciada com sucesso." if deleted else "Sessão não encontrada.",
    )


@app.get("/stats", response_model=SystemStats, tags=["Sistema"])
async def get_stats():
    """Retorna estatísticas detalhadas do sistema."""
    _require_chain()

    chain_stats = state.chain.get_stats()

    return SystemStats(
        model=chain_stats.get("model", MODEL_NAME),
        device=chain_stats.get("device", DEVICE),
        retrieval_mode=chain_stats.get("retrieval_mode", "hybrid"),
        vectorstore=chain_stats.get("vectorstore", {}),
        router=chain_stats.get("router", {}),
        memory=chain_stats.get("memory", {}),
        hybrid_retriever=chain_stats.get("hybrid_retriever", {}),
        active_sessions=len(state.sessions),
        uptime_seconds=round(state.uptime, 2),
    )
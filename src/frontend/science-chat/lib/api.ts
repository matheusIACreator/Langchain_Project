/**
 * lib/api.ts — Cliente HTTP para o FastAPI RAG Backend
 * Tipagens alinhadas com os modelos Pydantic do api.py
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ══════════════════════════════════════════════════════
// TIPOS
// ══════════════════════════════════════════════════════

export interface RoutingInfo {
    primary_expert: string;
    secondary_experts: string[];
    confidence: number;
    routing_reason: string;
}

export interface SourceDocument {
    content: string;
    source_collection: string;
    metadata: Record<string, unknown>;
}

export interface ChatResponse {
    answer: string;
    session_id: string;
    collections_used: string[];
    retrieval_mode: string;
    routing: RoutingInfo;
    source_documents: SourceDocument[];
    is_greeting: boolean;
    latency_ms: number;
}

export interface FigureInfo {
    key: string;
    name: string;
    period: string;
    collection: string;
    years: string;
    description: string;
}

export interface FiguresResponse {
    figures: FigureInfo[];
    total: number;
}

export interface HealthResponse {
    status: string;
    model_loaded: boolean;
    vectorstore_ready: boolean;
    active_sessions: number;
    uptime_seconds: number;
}

export interface ApiError {
    detail: string;
}

// ══════════════════════════════════════════════════════
// CLIENTE
// ══════════════════════════════════════════════════════

async function fetchApi<T>(
    path: string,
    options?: RequestInit
): Promise<T> {
    const res = await fetch(`${API_URL}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });

    if (!res.ok) {
        const err: ApiError = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
    }

    return res.json() as Promise<T>;
}

// ══════════════════════════════════════════════════════
// ENDPOINTS
// ══════════════════════════════════════════════════════

/**
 * Envia uma mensagem e recebe a resposta do sistema RAG.
 */
export async function sendMessage(
    message: string,
    sessionId?: string
): Promise<ChatResponse> {
    return fetchApi<ChatResponse>("/chat", {
        method: "POST",
        body: JSON.stringify({ message, session_id: sessionId }),
    });
}

/**
 * Lista as figuras históricas disponíveis.
 */
export async function getFigures(): Promise<FiguresResponse> {
    return fetchApi<FiguresResponse>("/figures");
}

/**
 * Verifica se a API está operacional.
 */
export async function getHealth(): Promise<HealthResponse> {
    return fetchApi<HealthResponse>("/health");
}

/**
 * Limpa o histórico de uma sessão.
 */
export async function clearConversation(sessionId: string): Promise<void> {
    await fetchApi(`/conversation/${sessionId}`, { method: "DELETE" });
}

// ══════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════

/**
 * Formata o nome da collection para exibição.
 * "renaissance/galileo_galilei" → "Galileo Galilei"
 */
export function formatCollection(collection: string): string {
    const figure = collection.split("/")[1] ?? collection;
    return figure
        .split("_")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
}

/**
 * Emoji por período histórico.
 */
export function periodEmoji(collection: string): string {
    if (collection.includes("renaissance")) return "🔭";
    if (collection.includes("enlightenment")) return "🍎";
    if (collection.includes("modern")) return "⚛️";
    return "📚";
}
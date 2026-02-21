"use client";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ChatResponse, formatCollection, periodEmoji } from "@/lib/api";

// ══════════════════════════════════════════════════════
// TIPOS
// ══════════════════════════════════════════════════════

export interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    metadata?: Pick<
        ChatResponse,
        "collections_used" | "routing" | "latency_ms" | "is_greeting" | "retrieval_mode"
    >;
}

// ══════════════════════════════════════════════════════
// COMPONENTE
// ══════════════════════════════════════════════════════

interface MessageBubbleProps {
    message: Message;
    showMetadata?: boolean;
}

export function MessageBubble({ message, showMetadata = true }: MessageBubbleProps) {
    const isUser = message.role === "user";

    return (
        <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
            {/* Avatar */}
            <Avatar className="h-8 w-8 shrink-0 mt-1">
                <AvatarFallback
                    className={
                        isUser
                            ? "bg-blue-600 text-white text-xs"
                            : "bg-slate-700 text-white text-xs"
                    }
                >
                    {isUser ? "TU" : "AI"}
                </AvatarFallback>
            </Avatar>

            {/* Conteúdo */}
            <div className={`flex flex-col gap-1 max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
                {/* Balão */}
                <div
                    className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${isUser
                            ? "bg-blue-600 text-white rounded-tr-sm"
                            : "bg-slate-800 text-slate-100 rounded-tl-sm"
                        }`}
                >
                    {message.content}
                </div>

                {/* Metadados — só para mensagens do assistente */}
                {!isUser && showMetadata && message.metadata && !message.metadata.is_greeting && (
                    <MetadataRow metadata={message.metadata} />
                )}
            </div>
        </div>
    );
}

// ── Linha de metadados ────────────────────────────────

function MetadataRow({
    metadata,
}: {
    metadata: NonNullable<Message["metadata"]>;
}) {
    const { collections_used, routing, latency_ms } = metadata;

    return (
        <div className="flex flex-wrap gap-1.5 px-1">
            {/* Collections usadas */}
            {collections_used?.map((col) => (
                <Badge
                    key={col}
                    variant="secondary"
                    className="text-xs bg-slate-700/60 text-slate-300 hover:bg-slate-700"
                >
                    {periodEmoji(col)} {formatCollection(col)}
                </Badge>
            ))}

            {/* Expert */}
            {routing?.primary_expert && routing.primary_expert !== "greeting" && (
                <Badge
                    variant="outline"
                    className="text-xs border-slate-600 text-slate-400"
                >
                    {routing.primary_expert}
                </Badge>
            )}

            {/* Latência */}
            {latency_ms != null && (
                <span className="text-xs text-slate-500 self-center">
                    {(latency_ms / 1000).toFixed(1)}s
                </span>
            )}
        </div>
    );
}
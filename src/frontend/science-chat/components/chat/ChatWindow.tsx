"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble, Message } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { sendMessage } from "@/lib/api";
import { Loader2 } from "lucide-react";

// ══════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════

function generateId() {
  return Math.random().toString(36).slice(2, 9);
}

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Olá! Sou um assistente especializado em cientistas históricos.\n\nPodes perguntar-me sobre **Galileu Galilei** 🔭, **Isaac Newton** 🍎 ou **Albert Einstein** ⚛️ — vida, descobertas, obras e legado.",
};

// ══════════════════════════════════════════════════════
// COMPONENTE
// ══════════════════════════════════════════════════════

export function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Scroll automático para o fim
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSend = useCallback(async (text: string) => {
    if (!text.trim()) return;

    // Adiciona mensagem do utilizador imediatamente
    const userMsg: Message = { id: generateId(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendMessage(text, sessionId);

      // Guarda session_id para manter contexto
      if (!sessionId) setSessionId(response.session_id);

      const assistantMsg: Message = {
        id: generateId(),
        role: "assistant",
        content: response.answer,
        metadata: {
          collections_used: response.collections_used,
          routing: response.routing,
          latency_ms: response.latency_ms,
          is_greeting: response.is_greeting,
          retrieval_mode: response.retrieval_mode,
        },
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Erro desconhecido";
      setError(msg);

      // Adiciona mensagem de erro inline
      setMessages((prev) => [
        ...prev,
        {
          id: generateId(),
          role: "assistant",
          content: `❌ ${msg}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  function handleStop() {
    abortRef.current?.abort();
    setIsLoading(false);
  }

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-2xl overflow-hidden border border-slate-700">
      {/* Mensagens */}
      <ScrollArea className="flex-1 px-4 py-4 overflow-y-auto" style={{ height: 0 }}>
        <div className="flex flex-col gap-4">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Indicador de loading */}
          {isLoading && (
            <div className="flex gap-3">
              <div className="h-8 w-8 shrink-0 rounded-full bg-slate-700 flex items-center justify-center mt-1">
                <Loader2 className="h-4 w-4 text-slate-400 animate-spin" />
              </div>
              <div className="bg-slate-800 rounded-2xl rounded-tl-sm px-4 py-2.5">
                <div className="flex gap-1 items-center h-5">
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        isLoading={isLoading}
        onStop={handleStop}
        disabled={false}
      />
    </div>
  );
}
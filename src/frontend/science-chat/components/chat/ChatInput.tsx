"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { SendHorizonal, Square } from "lucide-react";

interface ChatInputProps {
    onSend: (message: string) => void;
    isLoading: boolean;
    onStop?: () => void;
    disabled?: boolean;
}

export function ChatInput({ onSend, isLoading, onStop, disabled }: ChatInputProps) {
    const [value, setValue] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const canSend = value.trim().length > 0 && !isLoading && !disabled;

    function handleSend() {
        if (!canSend) return;
        onSend(value.trim());
        setValue("");
        textareaRef.current?.focus();
    }

    function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    }

    return (
        <div className="flex gap-2 items-end p-4 border-t border-slate-700 bg-slate-900">
            <Textarea
                ref={textareaRef}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Pergunte sobre Galileu, Newton ou Einstein… (Enter para enviar)"
                disabled={disabled}
                rows={1}
                className="resize-none min-h-[44px] max-h-[120px] bg-slate-800 border-slate-600 text-slate-100
                   placeholder:text-slate-500 focus-visible:ring-blue-500 rounded-xl"
                style={{ fieldSizing: "content" } as React.CSSProperties}
            />

            {isLoading ? (
                <Button
                    size="icon"
                    variant="destructive"
                    onClick={onStop}
                    className="shrink-0 h-11 w-11 rounded-xl"
                >
                    <Square className="h-4 w-4" />
                </Button>
            ) : (
                <Button
                    size="icon"
                    onClick={handleSend}
                    disabled={!canSend}
                    className="shrink-0 h-11 w-11 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-30"
                >
                    <SendHorizonal className="h-4 w-4" />
                </Button>
            )}
        </div>
    );
}
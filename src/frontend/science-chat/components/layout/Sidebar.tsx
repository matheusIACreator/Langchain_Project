"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { getFigures, FigureInfo, periodEmoji } from "@/lib/api";
import { Loader2 } from "lucide-react";

export function Sidebar() {
    const [figures, setFigures] = useState<FigureInfo[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        getFigures()
            .then((res) => setFigures(res.figures))
            .catch(() => setFigures([]))
            .finally(() => setLoading(false));
    }, []);

    return (
        <aside className="w-64 shrink-0 flex flex-col gap-4 p-4 bg-slate-900 border-r border-slate-700 h-full">
            {/* Logo */}
            <div className="flex flex-col gap-1">
                <h1 className="text-lg font-bold text-white tracking-tight">
                    🏛️ Science Chat
                </h1>
                <p className="text-xs text-slate-400">RAG Multi-Figura v2.0</p>
            </div>

            <Separator className="bg-slate-700" />

            {/* Figuras */}
            <div className="flex flex-col gap-2">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Figuras disponíveis
                </p>

                {loading ? (
                    <div className="flex items-center gap-2 text-slate-500 text-sm">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        A carregar…
                    </div>
                ) : (
                    figures.map((fig) => (
                        <FigureCard key={fig.key} figure={fig} />
                    ))
                )}
            </div>

            <Separator className="bg-slate-700" />

            {/* Dicas */}
            <div className="flex flex-col gap-2">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Exemplos
                </p>
                <ul className="flex flex-col gap-1.5 text-xs text-slate-400">
                    <li>• "Quando Newton nasceu?"</li>
                    <li>• "Compare Newton e Einstein"</li>
                    <li>• "Galileu e a Inquisição"</li>
                    <li>• "Einstein e o Nobel"</li>
                </ul>
            </div>
        </aside>
    );
}

function FigureCard({ figure }: { figure: FigureInfo }) {
    return (
        <Card className="bg-slate-800 border-slate-700 p-3 flex flex-col gap-1">
            <div className="flex items-center gap-2">
                <span className="text-base">{periodEmoji(figure.collection)}</span>
                <span className="text-sm font-medium text-white">{figure.name}</span>
            </div>
            <div className="flex gap-1 flex-wrap">
                <Badge
                    variant="secondary"
                    className="text-xs bg-slate-700 text-slate-300"
                >
                    {figure.period}
                </Badge>
                <Badge
                    variant="outline"
                    className="text-xs border-slate-600 text-slate-400"
                >
                    {figure.years}
                </Badge>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">{figure.description}</p>
        </Card>
    );
}
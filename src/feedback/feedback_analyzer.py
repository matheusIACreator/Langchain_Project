"""
Feedback Analyzer - Analisa e visualiza feedback coletado
Gera relatórios e insights para melhoria do sistema
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime
from collections import Counter

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from .feedback_collector import FeedbackCollector


class FeedbackAnalyzer:
    """
    Analisa feedback coletado e gera insights
    """
    
    def __init__(self, collector: FeedbackCollector = None):
        """
        Inicializa o analisador
        
        Args:
            collector: Coletor de feedback (opcional)
        """
        self.collector = collector or FeedbackCollector()
        print("📊 Feedback Analyzer inicializado")
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        Gera relatório completo de análise
        
        Args:
            output_file: Caminho para salvar o relatório (opcional)
            
        Returns:
            Dict com análises
        """
        print("\n" + "="*60)
        print("📊 GERANDO RELATÓRIO DE ANÁLISE")
        print("="*60 + "\n")
        
        # Estatísticas gerais
        stats = self.collector.get_feedback_stats()
        
        # Análise de sentimento (baseado em thumbs e ratings)
        sentiment = self._analyze_sentiment()
        
        # Análise de qualidade
        quality = self._analyze_quality()
        
        # Queries mais comuns
        common_queries = self._get_common_queries()
        
        # Problemas identificados
        issues = self._identify_issues()
        
        # Compilar relatório
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "sentiment_analysis": sentiment,
            "quality_analysis": quality,
            "common_queries": common_queries,
            "identified_issues": issues,
        }
        
        # Salvar se solicitado
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Relatório salvo em: {output_file}")
        
        # Imprimir resumo
        self._print_report(report)
        
        return report
    
    def _analyze_sentiment(self) -> Dict[str, Any]:
        """
        Analisa o sentimento geral dos feedbacks
        
        Returns:
            Dict com análise de sentimento
        """
        feedbacks = self.collector.get_all_feedbacks()
        
        if not feedbacks:
            return {"error": "Nenhum feedback disponível"}
        
        # Calcular sentimento baseado em thumbs e ratings
        positive = 0
        negative = 0
        neutral = 0
        
        for fb in feedbacks:
            if fb['thumbs_up'] is not None:
                if fb['thumbs_up']:
                    positive += 1
                else:
                    negative += 1
            elif fb['rating'] is not None:
                if fb['rating'] >= 4:
                    positive += 1
                elif fb['rating'] <= 2:
                    negative += 1
                else:
                    neutral += 1
        
        total = positive + negative + neutral
        
        if total == 0:
            return {"error": "Sem dados de sentimento"}
        
        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "positive_percentage": round(positive / total * 100, 1),
            "negative_percentage": round(negative / total * 100, 1),
            "neutral_percentage": round(neutral / total * 100, 1),
        }
    
    def _analyze_quality(self) -> Dict[str, Any]:
        """
        Analisa a qualidade das respostas
        
        Returns:
            Dict com análise de qualidade
        """
        feedbacks = self.collector.get_all_feedbacks()
        
        if not feedbacks:
            return {"error": "Nenhum feedback disponível"}
        
        # Feedbacks com rating
        rated = [fb for fb in feedbacks if fb['rating'] is not None]
        
        if not rated:
            return {"error": "Nenhum rating disponível"}
        
        ratings = [fb['rating'] for fb in rated]
        
        # Distribuição de ratings
        rating_dist = Counter(ratings)
        
        # Qualidade por nível
        excellent = sum(1 for r in ratings if r == 5)
        good = sum(1 for r in ratings if r == 4)
        average = sum(1 for r in ratings if r == 3)
        poor = sum(1 for r in ratings if r == 2)
        bad = sum(1 for r in ratings if r == 1)
        
        total = len(ratings)
        
        return {
            "average_rating": round(sum(ratings) / total, 2),
            "total_rated": total,
            "rating_distribution": dict(rating_dist),
            "quality_levels": {
                "excellent (5)": excellent,
                "good (4)": good,
                "average (3)": average,
                "poor (2)": poor,
                "bad (1)": bad,
            },
            "quality_percentages": {
                "excellent": round(excellent / total * 100, 1),
                "good": round(good / total * 100, 1),
                "average": round(average / total * 100, 1),
                "poor": round(poor / total * 100, 1),
                "bad": round(bad / total * 100, 1),
            }
        }
    
    def _get_common_queries(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Identifica as queries mais comuns
        
        Args:
            top_n: Número de queries a retornar
            
        Returns:
            Lista de queries mais comuns
        """
        feedbacks = self.collector.get_all_feedbacks()
        
        if not feedbacks:
            return []
        
        # Contar queries
        query_counter = Counter()
        query_ratings = {}
        
        for fb in feedbacks:
            query = fb['query'].lower().strip()
            query_counter[query] += 1
            
            # Armazenar ratings
            if fb['rating'] is not None:
                if query not in query_ratings:
                    query_ratings[query] = []
                query_ratings[query].append(fb['rating'])
        
        # Top queries com estatísticas
        common = []
        for query, count in query_counter.most_common(top_n):
            item = {
                "query": query,
                "count": count,
            }
            
            if query in query_ratings:
                ratings = query_ratings[query]
                item["avg_rating"] = round(sum(ratings) / len(ratings), 2)
                item["min_rating"] = min(ratings)
                item["max_rating"] = max(ratings)
            
            common.append(item)
        
        return common
    
    def _identify_issues(self) -> Dict[str, Any]:
        """
        Identifica problemas potenciais baseado em feedback negativo
        
        Returns:
            Dict com problemas identificados
        """
        # Pegar feedbacks negativos (thumbs down ou rating <= 2)
        negative_feedbacks = self.collector.get_all_feedbacks(min_rating=0)
        negative_feedbacks = [
            fb for fb in negative_feedbacks 
            if (fb['thumbs_up'] == False) or (fb['rating'] and fb['rating'] <= 2)
        ]
        
        if not negative_feedbacks:
            return {"message": "Nenhum problema identificado! 🎉"}
        
        # Análise de problemas
        issues = {
            "total_negative": len(negative_feedbacks),
            "queries_with_issues": [],
            "common_complaints": [],
        }
        
        # Queries problemáticas
        for fb in negative_feedbacks[:10]:  # Top 10
            issue = {
                "query": fb['query'],
                "response": fb['response'][:100] + "...",
                "rating": fb['rating'],
                "comment": fb['comment']
            }
            issues["queries_with_issues"].append(issue)
        
        # Análise de comentários (palavras-chave negativas)
        negative_keywords = [
            'errado', 'incorreto', 'ruim', 'não respondeu', 'confuso',
            'incompleto', 'impreciso', 'vago', 'genérico'
        ]
        
        for fb in negative_feedbacks:
            if fb['comment']:
                comment_lower = fb['comment'].lower()
                for keyword in negative_keywords:
                    if keyword in comment_lower:
                        issues["common_complaints"].append({
                            "keyword": keyword,
                            "comment": fb['comment']
                        })
        
        return issues
    
    def _print_report(self, report: Dict[str, Any]) -> None:
        """
        Imprime relatório formatado no console
        
        Args:
            report: Dict com o relatório
        """
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE ANÁLISE DE FEEDBACK")
        print("="*60)
        
        # Estatísticas gerais
        stats = report["statistics"]
        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"   Total de feedbacks: {stats['total_feedbacks']}")
        print(f"   Com rating: {stats['with_rating']}")
        print(f"   Rating médio: {stats['avg_rating']}/5.0")
        print(f"   👍 Thumbs up: {stats['thumbs_up']}")
        print(f"   👎 Thumbs down: {stats['thumbs_down']}")
        print(f"   Comentários: {stats['with_comment']}")
        
        # Sentimento
        if "error" not in report["sentiment_analysis"]:
            sentiment = report["sentiment_analysis"]
            print(f"\n😊 ANÁLISE DE SENTIMENTO:")
            print(f"   Positivo: {sentiment['positive']} ({sentiment['positive_percentage']}%)")
            print(f"   Negativo: {sentiment['negative']} ({sentiment['negative_percentage']}%)")
            print(f"   Neutro: {sentiment['neutral']} ({sentiment['neutral_percentage']}%)")
        
        # Qualidade
        if "error" not in report["quality_analysis"]:
            quality = report["quality_analysis"]
            print(f"\n⭐ ANÁLISE DE QUALIDADE:")
            print(f"   Rating médio: {quality['average_rating']}/5.0")
            print(f"   Total avaliado: {quality['total_rated']}")
            print(f"\n   Distribuição:")
            for level, count in quality["quality_levels"].items():
                percentage = quality["quality_percentages"][level.split()[0]]
                print(f"      {level}: {count} ({percentage}%)")
        
        # Queries comuns
        if report["common_queries"]:
            print(f"\n🔥 TOP 5 QUERIES MAIS COMUNS:")
            for i, item in enumerate(report["common_queries"][:5], 1):
                query_text = item['query'][:50] + "..." if len(item['query']) > 50 else item['query']
                print(f"   {i}. \"{query_text}\"")
                print(f"      Frequência: {item['count']}x", end="")
                if 'avg_rating' in item:
                    print(f", Rating médio: {item['avg_rating']}/5.0")
                else:
                    print()
        
        # Problemas
        issues = report["identified_issues"]
        if "message" in issues:
            print(f"\n✅ {issues['message']}")
        else:
            print(f"\n⚠️  PROBLEMAS IDENTIFICADOS:")
            print(f"   Total de feedbacks negativos: {issues['total_negative']}")
            
            if issues["queries_with_issues"]:
                print(f"\n   Queries problemáticas (top 3):")
                for i, issue in enumerate(issues["queries_with_issues"][:3], 1):
                    print(f"      {i}. \"{issue['query'][:40]}...\"")
                    if issue['comment']:
                        print(f"         Comentário: {issue['comment'][:60]}...")
        
        print("\n" + "="*60 + "\n")
    
    def export_insights_for_improvement(self, output_file: str) -> None:
        """
        Exporta insights específicos para melhoria do sistema
        
        Args:
            output_file: Arquivo de saída
        """
        report = self.generate_report()
        
        # Compilar insights acionáveis
        insights = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": [],
        }
        
        # Recomendações baseadas em qualidade
        quality = report.get("quality_analysis", {})
        if "average_rating" in quality:
            avg_rating = quality["average_rating"]
            
            if avg_rating < 3.5:
                insights["recommendations"].append({
                    "priority": "HIGH",
                    "category": "quality",
                    "issue": f"Rating médio baixo ({avg_rating}/5.0)",
                    "action": "Revisar prompts e melhorar retrieval"
                })
            
            # Muito ratings baixos
            poor_pct = quality.get("quality_percentages", {}).get("poor", 0)
            bad_pct = quality.get("quality_percentages", {}).get("bad", 0)
            
            if (poor_pct + bad_pct) > 20:
                insights["recommendations"].append({
                    "priority": "HIGH",
                    "category": "quality",
                    "issue": f"{poor_pct + bad_pct}% de respostas ruins",
                    "action": "Considerar fine-tuning com DPO"
                })
        
        # Recomendações baseadas em problemas
        issues = report.get("identified_issues", {})
        if issues.get("total_negative", 0) > 10:
            insights["recommendations"].append({
                "priority": "MEDIUM",
                "category": "user_satisfaction",
                "issue": f"{issues['total_negative']} feedbacks negativos",
                "action": "Analisar queries problemáticas e ajustar"
            })
        
        # Recomendações baseadas em volume
        stats = report.get("statistics", {})
        if stats.get("total_feedbacks", 0) >= 500:
            insights["recommendations"].append({
                "priority": "MEDIUM",
                "category": "training",
                "issue": "Volume suficiente de dados",
                "action": "Considerar treinar modelo com DPO"
            })
        
        # Salvar
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Insights exportados para: {output_file}")
        
        # Imprimir recomendações
        print("\n" + "="*60)
        print("💡 RECOMENDAÇÕES DE MELHORIA")
        print("="*60)
        
        if not insights["recommendations"]:
            print("\n✅ Sistema funcionando bem! Continue coletando feedback.")
        else:
            for rec in insights["recommendations"]:
                priority_emoji = "🔴" if rec["priority"] == "HIGH" else "🟡"
                print(f"\n{priority_emoji} [{rec['priority']}] {rec['category'].upper()}")
                print(f"   Problema: {rec['issue']}")
                print(f"   Ação: {rec['action']}")
        
        print("\n" + "="*60 + "\n")


def main():
    """
    Teste standalone do analisador
    """
    print("\n" + "="*60)
    print("🧪 TESTANDO FEEDBACK ANALYZER")
    print("="*60 + "\n")
    
    # Inicializar
    analyzer = FeedbackAnalyzer()
    
    # Gerar relatório
    report = analyzer.generate_report("data/feedback/analysis_report.json")
    
    # Exportar insights
    analyzer.export_insights_for_improvement("data/feedback/improvement_insights.json")
    
    print("\n✅ Análise concluída!")


if __name__ == "__main__":
    main()
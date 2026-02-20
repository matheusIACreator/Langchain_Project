"""
Topic Router - Sistema de roteamento inteligente
Direciona queries para os experts e collections apropriados
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.experts_config import EXPERTS, ROUTING_RULES, ROUTING_THRESHOLD, DEFAULT_EXPERT


class TopicRouter:
    """
    Roteador que analisa queries e determina quais experts/collections usar
    """
    FIGURE_TO_PERIOD = {
    'galileo_galilei': 'renaissance',
    'isaac_newton': 'enlightenment',
    'albert_einstein': 'modern_era',
    'leonardo_da_vinci': 'renaissance',
    'marie_curie': 'modern_era',
    'charles_darwin': 'modern_era',
}
    def __init__(self):
        """
        Inicializa o roteador
        """
        print("🧭 Inicializando Topic Router...")
        
        # Compilar regras de routing para eficiência
        self.routing_rules = ROUTING_RULES
        self.experts = EXPERTS
        
        # Cache de routing (para queries similares)
        self.routing_cache = {}
        
        print("✅ Topic Router inicializado!")
    
    def route_query(self, query: str) -> Dict[str, any]:
        """
        Analisa query e retorna roteamento apropriado
        
        Args:
            query: Query do usuário
            
        Returns:
            Dict com:
            - primary_expert: Expert principal
            - secondary_experts: Experts secundários (se houver)
            - confidence: Confiança na decisão (0-1)
            - routing_reason: Razão do roteamento
        """
        query_lower = query.lower()
        
        # Verificar cache
        if query_lower in self.routing_cache:
            return self.routing_cache[query_lower]
        
        # Calcular scores para cada expert
        expert_scores = self._calculate_expert_scores(query_lower)
        
        # Determinar experts a usar
        routing = self._determine_routing(expert_scores, query)
        
        # Cachear resultado
        self.routing_cache[query_lower] = routing
        
        return routing
    
    def _calculate_expert_scores(self, query: str) -> Dict[str, float]:
        """
        Calcula score de relevância de cada expert para a query
        
        Args:
            query: Query em lowercase
            
        Returns:
            Dict com scores por expert
        """
        scores = {}
        
        for expert_name, keywords in self.routing_rules.items():
            # Contar quantas keywords aparecem na query
            matches = sum(1 for kw in keywords if kw in query)
            
            # Score normalizado
            if matches > 0:
                scores[expert_name] = min(matches / 3.0, 1.0)  # Max 1.0
            else:
                scores[expert_name] = 0.0
        
        return scores
    
    def _determine_routing(self, scores: Dict[str, float], query: str) -> Dict:
        """
        Determina o routing final baseado nos scores
        
        Args:
            scores: Scores dos experts
            query: Query original
            
        Returns:
            Decisão de routing
        """
        # Ordenar experts por score
        sorted_experts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Expert principal (maior score)
        primary_expert = sorted_experts[0][0] if sorted_experts[0][1] >= ROUTING_THRESHOLD else DEFAULT_EXPERT
        primary_score = sorted_experts[0][1] if sorted_experts[0][1] >= ROUTING_THRESHOLD else 0.5
        
        # Experts secundários (score > threshold)
        secondary_experts = [
            expert for expert, score in sorted_experts[1:]
            if score >= ROUTING_THRESHOLD
        ]
        
        # Determinar razão do roteamento
        routing_reason = self._get_routing_reason(primary_expert, primary_score, query)
        
        routing = {
            'primary_expert': primary_expert,
            'secondary_experts': secondary_experts[:2],  # Max 2 secundários
            'confidence': primary_score,
            'routing_reason': routing_reason,
            'all_scores': scores
        }
        
        return routing
    
    def _get_routing_reason(self, expert: str, score: float, query: str) -> str:
        """
        Gera explicação legível do roteamento
        
        Args:
            expert: Expert selecionado
            score: Score de confiança
            query: Query original
            
        Returns:
            Texto explicativo
        """
        if score < ROUTING_THRESHOLD:
            return f"Query genérica, usando expert padrão ({DEFAULT_EXPERT})"
        
        expert_config = self.experts.get(expert)
        if expert_config:
            return f"Roteado para {expert_config.name} (confiança: {score:.2f})"
        
        return f"Roteado para {expert} (confiança: {score:.2f})"
    
    def route_to_collections(self, query: str, available_periods: List[str]) -> List[str]:
        query_lower = query.lower()
        mentioned_figures = self._detect_figures(query_lower)
        mentioned_periods = self._detect_periods(query_lower)

        if mentioned_figures:
            collections = []
            for figure in mentioned_figures:
                period = self.FIGURE_TO_PERIOD.get(figure)
                if period and f"{period}/{figure}" not in collections:
                    collections.append(f"{period}/{figure}")
            if collections:
                return collections

        if mentioned_periods:
            return mentioned_periods

        return available_periods
    
    def _detect_figures(self, query: str) -> List[str]:
        """
        Detecta menções a figuras históricas na query
        
        Args:
            query: Query em lowercase
            
        Returns:
            Lista de figuras detectadas
        """
        # Mapeamento de variações de nomes
        figure_variations = {
            'galileo_galilei': ['galileu', 'galilei', 'galileo'],
            'isaac_newton': ['newton', 'isaac'],
            'albert_einstein': ['einstein', 'albert'],
            'leonardo_da_vinci': ['leonardo', 'da vinci', 'vinci'],
            'marie_curie': ['curie', 'marie'],
            'charles_darwin': ['darwin', 'charles'],
        }
        
        detected = []
        
        for figure, variations in figure_variations.items():
            if any(var in query for var in variations):
                detected.append(figure)
        
        return detected
    
    def _detect_periods(self, query: str) -> List[str]:
        """
        Detecta menções a períodos históricos na query
        
        Args:
            query: Query em lowercase
            
        Returns:
            Lista de períodos detectados
        """
        period_keywords = {
            'renaissance': ['renascimento', 'renascença', 'renascentista', 'século 15', 'século 16'],
            'enlightenment': ['iluminismo', 'ilustração', 'século 17', 'século 18'],
            'modern_era': ['era moderna', 'século 19', 'século 20', 'contemporâneo'],
        }
        
        detected = []
        
        for period, keywords in period_keywords.items():
            if any(kw in query for kw in keywords):
                detected.append(period)
        
        return detected
    
    def get_routing_stats(self) -> Dict:
        """
        Retorna estatísticas do roteamento
        
        Returns:
            Dict com estatísticas
        """
        if not self.routing_cache:
            return {
                'total_queries': 0,
                'most_used_expert': None,
                'cache_size': 0
            }
        
        # Contar uso de cada expert
        expert_usage = Counter([
            routing['primary_expert']
            for routing in self.routing_cache.values()
        ])
        
        most_used = expert_usage.most_common(1)[0] if expert_usage else ('None', 0)
        
        return {
            'total_queries': len(self.routing_cache),
            'most_used_expert': most_used[0],
            'expert_usage': dict(expert_usage),
            'cache_size': len(self.routing_cache)
        }
    
    def clear_cache(self):
        """
        Limpa o cache de roteamento
        """
        self.routing_cache.clear()
        print("🗑️  Cache de roteamento limpo!")


def main():
    """
    Função principal para teste standalone
    """
    print("\n" + "="*60)
    print("🧪 TESTANDO TOPIC ROUTER")
    print("="*60 + "\n")
    
    # Inicializar router
    router = TopicRouter()
    
    # Queries de teste
    test_queries = [
        "Quando Galileu nasceu?",
        "Explique as leis do movimento de Newton",
        "O que Galileu descobriu com o telescópio?",
        "Como a Igreja reagiu às ideias de Galileu?",
        "Qual era o método científico de Galileu?",
        "Compare as teorias de Newton e Einstein sobre gravidade",
        "Quem foi Leonardo da Vinci?",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        
        # Roteamento
        routing = router.route_query(query)
        
        print(f"\n📍 Roteamento:")
        print(f"   Expert Principal: {routing['primary_expert']}")
        if routing['secondary_experts']:
            print(f"   Experts Secundários: {', '.join(routing['secondary_experts'])}")
        print(f"   Confiança: {routing['confidence']:.2f}")
        print(f"   Razão: {routing['routing_reason']}")
        
        # Collections
        collections = router.route_to_collections(query, ['renaissance', 'enlightenment', 'modern_era'])
        print(f"\n📚 Collections a buscar:")
        for col in collections:
            print(f"   - {col}")
    
    # Estatísticas
    print(f"\n{'='*60}")
    print("📊 ESTATÍSTICAS DO ROUTER")
    print(f"{'='*60}")
    stats = router.get_routing_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Teste concluído!")


if __name__ == "__main__":
    main()

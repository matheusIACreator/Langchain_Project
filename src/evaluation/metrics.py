"""
Evaluation Metrics - Sistema de métricas para avaliar qualidade das respostas
Implementa métricas de precisão, relevância, citação e detecção de alucinações
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.documents import Document


class EvaluationMetrics:
    """
    Sistema de métricas para avaliar qualidade das respostas RAG
    """
    
    def __init__(self):
        """Inicializa o sistema de métricas"""
        print("📊 Inicializando sistema de métricas...")
        
        # Padrões para detecção
        self.date_pattern = r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b|\b\d{4}\b'
        self.name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        
        print("✅ Sistema de métricas inicializado!")
    
    def evaluate_answer(
        self,
        query: str,
        answer: str,
        source_documents: List[Document],
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Avalia uma resposta completa
        
        Args:
            query: Pergunta original
            answer: Resposta gerada
            source_documents: Documentos usados como contexto
            expected_answer: Resposta esperada (opcional, para comparação)
            
        Returns:
            Dict com scores e análise detalhada
        """
        print(f"\n{'='*60}")
        print(f"📊 AVALIANDO RESPOSTA")
        print(f"{'='*60}")
        
        # Executar todas as métricas
        relevance = self.calculate_relevance(query, answer)
        groundedness = self.calculate_groundedness(answer, source_documents)
        citation_quality = self.evaluate_citation_quality(answer, source_documents)
        factual_consistency = self.check_factual_consistency(answer, source_documents)
        hallucination_score = self.detect_hallucination(answer, source_documents)
        
        # Score agregado
        overall_score = (
            relevance * 0.25 +
            groundedness * 0.25 +
            citation_quality * 0.20 +
            factual_consistency * 0.20 +
            (1 - hallucination_score) * 0.10
        )
        
        # Análise detalhada
        analysis = {
            "overall_score": round(overall_score, 3),
            "metrics": {
                "relevance": round(relevance, 3),
                "groundedness": round(groundedness, 3),
                "citation_quality": round(citation_quality, 3),
                "factual_consistency": round(factual_consistency, 3),
                "hallucination_score": round(hallucination_score, 3),
            },
            "details": {
                "query_length": len(query.split()),
                "answer_length": len(answer.split()),
                "sources_used": len(source_documents),
                "dates_mentioned": len(re.findall(self.date_pattern, answer)),
                "names_mentioned": len(re.findall(self.name_pattern, answer)),
            },
            "quality_grade": self._get_quality_grade(overall_score),
            "issues": self._identify_issues(
                relevance, groundedness, citation_quality,
                factual_consistency, hallucination_score
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        # Comparar com resposta esperada se fornecida
        if expected_answer:
            analysis["expected_match"] = self.compare_with_expected(
                answer, expected_answer
            )
        
        self._print_evaluation_summary(analysis)
        
        return analysis
    
    def calculate_relevance(self, query: str, answer: str) -> float:
        """
        Calcula relevância da resposta em relação à query
        
        Usa overlap de palavras-chave e análise de tópicos
        """
        # Extrair palavras-chave (removendo stopwords simples)
        stopwords = {'o', 'a', 'de', 'da', 'do', 'que', 'e', 'é', 'em', 'para', 'com', 'por', 'um', 'uma'}
        
        query_words = set(query.lower().split()) - stopwords
        answer_words = set(answer.lower().split()) - stopwords
        
        if not query_words:
            return 0.0
        
        # Overlap de palavras
        overlap = len(query_words & answer_words)
        relevance = overlap / len(query_words)
        
        # Boost se resposta contém entidades da query
        entities_query = set(re.findall(self.name_pattern, query))
        entities_answer = set(re.findall(self.name_pattern, answer))
        
        if entities_query:
            entity_match = len(entities_query & entities_answer) / len(entities_query)
            relevance = (relevance + entity_match) / 2
        
        return min(relevance, 1.0)
    
    def calculate_groundedness(
        self,
        answer: str,
        source_documents: List[Document]
    ) -> float:
        """
        Calcula quanto da resposta está fundamentada nos documentos fonte
        
        Verifica se afirmações aparecem nos documentos
        """
        if not source_documents:
            return 0.0
        
        # Combinar todo o contexto
        all_context = " ".join([doc.page_content.lower() for doc in source_documents])
        
        # Dividir resposta em sentenças
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        grounded_count = 0
        
        for sentence in sentences:
            # Extrair palavras-chave da sentença
            words = set(sentence.lower().split())
            
            # Remover stopwords
            stopwords = {'o', 'a', 'de', 'da', 'do', 'que', 'e', 'é', 'foi', 'em', 'para'}
            keywords = words - stopwords
            
            if not keywords:
                continue
            
            # Verificar quantas keywords aparecem no contexto
            matches = sum(1 for kw in keywords if kw in all_context)
            
            # Se >50% das keywords estão no contexto, considerar grounded
            if matches / len(keywords) > 0.5:
                grounded_count += 1
        
        return grounded_count / len(sentences)
    
    def evaluate_citation_quality(
        self,
        answer: str,
        source_documents: List[Document]
    ) -> float:
        """
        Avalia qualidade das citações (se mencionam fontes, períodos, etc)
        """
        score = 0.0
        
        # Verificar se menciona fontes/contexto
        if any(keyword in answer.lower() for keyword in ['segundo', 'de acordo', 'conforme', 'baseado']):
            score += 0.3
        
        # Verificar se menciona períodos/datas (indica precisão histórica)
        dates = re.findall(self.date_pattern, answer)
        if dates:
            score += 0.3
        
        # Verificar se menciona nomes próprios (indica especificidade)
        names = re.findall(self.name_pattern, answer)
        if names:
            score += 0.2
        
        # Verificar se usa fontes fornecidas
        if source_documents:
            # Extrair figuras dos metadados
            figures = set()
            for doc in source_documents:
                if 'figure' in doc.metadata:
                    figures.add(doc.metadata['figure'])
            
            # Verificar se menciona as figuras relevantes
            for figure in figures:
                figure_name = figure.replace('_', ' ').title()
                if figure_name.lower() in answer.lower():
                    score += 0.2
                    break
        
        return min(score, 1.0)
    
    def check_factual_consistency(
        self,
        answer: str,
        source_documents: List[Document]
    ) -> float:
        """
        Verifica consistência factual básica
        
        Checa se datas e nomes na resposta aparecem nos documentos
        """
        if not source_documents:
            return 0.5  # Neutro se não há contexto
        
        # Combinar contexto
        all_context = " ".join([doc.page_content for doc in source_documents])
        
        # Extrair datas da resposta
        answer_dates = set(re.findall(self.date_pattern, answer))
        
        # Extrair nomes da resposta
        answer_names = set(re.findall(self.name_pattern, answer))
        
        # Extrair do contexto
        context_dates = set(re.findall(self.date_pattern, all_context))
        context_names = set(re.findall(self.name_pattern, all_context))
        
        consistency_score = 1.0
        
        # Penalizar datas que não aparecem no contexto
        if answer_dates:
            date_overlap = len(answer_dates & context_dates) / len(answer_dates)
            consistency_score *= (0.5 + 0.5 * date_overlap)
        
        # Penalizar nomes que não aparecem no contexto
        if answer_names:
            name_overlap = len(answer_names & context_names) / len(answer_names)
            consistency_score *= (0.5 + 0.5 * name_overlap)
        
        return consistency_score
    
    def detect_hallucination(
        self,
        answer: str,
        source_documents: List[Document]
    ) -> float:
        """
        Detecta possíveis alucinações
        
        Score mais alto = mais provável alucinação
        """
        if not source_documents:
            return 1.0  # Sem contexto = alta chance de alucinação
        
        hallucination_indicators = 0
        
        # Indicador 1: Muitas informações específicas não no contexto
        all_context = " ".join([doc.page_content.lower() for doc in source_documents])
        
        # Frases de incerteza (bom - indica que não está inventando)
        uncertainty_phrases = ['aproximadamente', 'cerca de', 'possivelmente', 
                              'provavelmente', 'pode ter', 'talvez']
        
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        if not has_uncertainty and len(answer.split()) > 50:
            # Resposta longa e assertiva sem incerteza = possível alucinação
            hallucination_indicators += 0.3
        
        # Indicador 2: Números/datas muito específicos não no contexto
        specific_numbers = re.findall(r'\b\d+\b', answer)
        if specific_numbers:
            numbers_in_context = sum(1 for num in specific_numbers if num in all_context)
            if len(specific_numbers) > 0:
                if numbers_in_context / len(specific_numbers) < 0.3:
                    hallucination_indicators += 0.3
        
        # Indicador 3: Contradiz informação do contexto
        # (simplificado - detecta negações)
        if 'não' in answer.lower() or 'nunca' in answer.lower():
            hallucination_indicators += 0.2
        
        return min(hallucination_indicators, 1.0)
    
    def compare_with_expected(
        self,
        answer: str,
        expected_answer: str
    ) -> Dict[str, Any]:
        """
        Compara resposta gerada com resposta esperada (ground truth)
        """
        # Palavras em comum
        answer_words = set(answer.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        overlap = len(answer_words & expected_words)
        precision = overlap / len(answer_words) if answer_words else 0
        recall = overlap / len(expected_words) if expected_words else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Similaridade estrutural simples
        answer_entities = set(re.findall(self.name_pattern, answer))
        expected_entities = set(re.findall(self.name_pattern, expected_answer))
        
        entity_match = len(answer_entities & expected_entities) / len(expected_entities) if expected_entities else 0
        
        return {
            "word_overlap": round(overlap / max(len(answer_words), len(expected_words)), 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "entity_match": round(entity_match, 3)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Converte score em grade qualitativa"""
        if score >= 0.9:
            return "Excelente (A)"
        elif score >= 0.8:
            return "Muito Bom (B)"
        elif score >= 0.7:
            return "Bom (C)"
        elif score >= 0.6:
            return "Satisfatório (D)"
        else:
            return "Insuficiente (F)"
    
    def _identify_issues(
        self,
        relevance: float,
        groundedness: float,
        citation_quality: float,
        factual_consistency: float,
        hallucination_score: float
    ) -> List[str]:
        """Identifica problemas específicos"""
        issues = []
        
        if relevance < 0.5:
            issues.append("⚠️ Baixa relevância - resposta não aborda bem a pergunta")
        
        if groundedness < 0.5:
            issues.append("⚠️ Baixo grounding - resposta não está bem fundamentada no contexto")
        
        if citation_quality < 0.4:
            issues.append("⚠️ Citações fracas - faltam referências específicas")
        
        if factual_consistency < 0.6:
            issues.append("⚠️ Inconsistências factuais - datas/nomes podem estar incorretos")
        
        if hallucination_score > 0.5:
            issues.append("🚨 Possível alucinação - resposta pode conter informações inventadas")
        
        if not issues:
            issues.append("✅ Nenhum problema crítico detectado")
        
        return issues
    
    def _print_evaluation_summary(self, analysis: Dict[str, Any]):
        """Imprime resumo da avaliação"""
        print(f"\n{'='*60}")
        print("📊 RESULTADO DA AVALIAÇÃO")
        print(f"{'='*60}")
        
        print(f"\n🎯 Score Geral: {analysis['overall_score']:.3f} - {analysis['quality_grade']}")
        
        print(f"\n📈 Métricas Individuais:")
        for metric, score in analysis['metrics'].items():
            bar = '█' * int(score * 20)
            print(f"   {metric:.<25} {score:.3f} {bar}")
        
        print(f"\n📋 Detalhes:")
        for key, value in analysis['details'].items():
            print(f"   {key}: {value}")
        
        print(f"\n⚠️  Issues Identificados:")
        for issue in analysis['issues']:
            print(f"   {issue}")
        
        if 'expected_match' in analysis:
            print(f"\n🎯 Comparação com Ground Truth:")
            for key, value in analysis['expected_match'].items():
                print(f"   {key}: {value}")
        
        print(f"{'='*60}\n")


def main():
    """Teste standalone"""
    print("\n" + "="*60)
    print("🧪 TESTANDO SISTEMA DE MÉTRICAS")
    print("="*60)
    
    # Mock data para teste
    query = "Quando Galileu Galilei nasceu?"
    
    answer = "Galileu Galilei nasceu em 15 de fevereiro de 1564, em Pisa, Itália."
    
    mock_doc = Document(
        page_content="Galileo Galilei was born on 15 February 1564 in Pisa, Italy. He was an Italian astronomer, physicist and engineer.",
        metadata={'figure': 'galileo_galilei', 'period': 'renaissance'}
    )
    
    expected_answer = "Galileu nasceu em 1564 em Pisa."
    
    # Avaliar
    evaluator = EvaluationMetrics()
    
    result = evaluator.evaluate_answer(
        query=query,
        answer=answer,
        source_documents=[mock_doc],
        expected_answer=expected_answer
    )
    
    print("\n✅ Teste concluído!")
    print(f"Score final: {result['overall_score']:.3f}")


if __name__ == "__main__":
    main()
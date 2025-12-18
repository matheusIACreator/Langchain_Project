"""
Expert Configuration - Configuração dos Experts especializados
Define os modelos especializados e suas configurações
"""

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ExpertConfig:
    """
    Configuração de um expert especializado
    """
    name: str
    model_path: str
    specialization: str
    topics: List[str]
    periods: List[str]
    load_priority: int  # 1 = alta, 3 = baixa
    description: str


# ===== CONFIGURAÇÕES DOS EXPERTS =====

EXPERTS = {
    # Expert em Física
    'physics': ExpertConfig(
        name="Physics Expert",
        model_path="experts/physics_expert",
        specialization="physics",
        topics=["mechanics", "gravity", "optics", "thermodynamics", "relativity", "quantum"],
        periods=["renaissance", "enlightenment", "modern_era"],
        load_priority=1,
        description="Especializado em conceitos físicos e descobertas científicas"
    ),
    
    # Expert em Biografia
    'biography': ExpertConfig(
        name="Biography Expert",
        model_path="experts/biography_expert",
        specialization="biography",
        topics=["birth", "death", "education", "family", "career", "achievements"],
        periods=["all"],
        load_priority=2,
        description="Especializado em eventos biográficos e cronologia pessoal"
    ),
    
    # Expert em Astronomia
    'astronomy': ExpertConfig(
        name="Astronomy Expert",
        model_path="experts/astronomy_expert",
        specialization="astronomy",
        topics=["planets", "stars", "telescopes", "observations", "solar_system"],
        periods=["renaissance", "enlightenment", "modern_era"],
        load_priority=1,
        description="Especializado em astronomia e observações celestes"
    ),
    
    # Expert em Filosofia
    'philosophy': ExpertConfig(
        name="Philosophy Expert",
        model_path="experts/philosophy_expert",
        specialization="philosophy",
        topics=["epistemology", "scientific_method", "ethics", "metaphysics"],
        periods=["enlightenment", "modern_era"],
        load_priority=3,
        description="Especializado em filosofia da ciência e pensamento crítico"
    ),
    
    # Expert em Contexto Histórico
    'historical_context': ExpertConfig(
        name="Historical Context Expert",
        model_path="experts/historical_expert",
        specialization="history",
        topics=["wars", "politics", "church", "society", "culture"],
        periods=["all"],
        load_priority=2,
        description="Especializado em contexto histórico e social"
    ),
}


# ===== REGRAS DE ROUTING =====

ROUTING_RULES = {
    # Keywords que indicam cada expert
    'physics': [
        'física', 'fisica', 'lei', 'força', 'forca', 'movimento',
        'gravidade', 'mecânica', 'mecanica', 'energia', 'inércia', 'inercia'
    ],
    
    'biography': [
        'nasceu', 'morreu', 'faleceu', 'vida', 'infância', 'infancia',
        'educação', 'educacao', 'família', 'familia', 'juventude',
        'quando', 'onde', 'idade'
    ],
    
    'astronomy': [
        'lua', 'sol', 'planeta', 'estrela', 'telescópio', 'telescopio',
        'júpiter', 'jupiter', 'vênus', 'venus', 'marte', 'saturno',
        'observação', 'observacao', 'céu', 'ceu', 'constelação', 'constelacao'
    ],
    
    'philosophy': [
        'método', 'metodo', 'científico', 'cientifico', 'pensamento',
        'razão', 'razao', 'lógica', 'logica', 'verdade', 'conhecimento',
        'epistemologia', 'filosofia'
    ],
    
    'historical_context': [
        'igreja', 'papa', 'inquisição', 'inquisicao', 'julgamento',
        'política', 'politica', 'guerra', 'sociedade', 'época', 'epoca',
        'século', 'seculo', 'período', 'periodo', 'contexto'
    ],
}


# ===== THRESHOLDS E CONFIGS =====

# Score mínimo para considerar um expert relevante
ROUTING_THRESHOLD = 0.3

# Número máximo de experts a carregar simultaneamente
MAX_LOADED_EXPERTS = 2

# Configuração de memória para experts
EXPERT_MEMORY_CONFIG = {
    'use_quantization': True,
    'quantization_bits': 4,
    'offload_to_cpu': True,  # Offload quando não em uso
}


# ===== FALLBACK CONFIG =====

# Expert padrão quando não há match claro
DEFAULT_EXPERT = 'biography'

# Expert para queries gerais
GENERAL_EXPERT = 'biography'

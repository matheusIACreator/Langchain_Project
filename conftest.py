"""
conftest.py — Configuração global do pytest
Deve ficar na raiz do projeto (ao lado de tests/)
"""

import pytest


def pytest_configure(config):
    """Registra marcadores customizados."""
    config.addinivalue_line(
        "markers",
        "llm: testes que requerem LLM carregado (GPU/CPU). "
        "Use -m 'not llm' para pular.",
    )
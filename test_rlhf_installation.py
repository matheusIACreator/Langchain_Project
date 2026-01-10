"""
Script de verificação da instalação RLHF
Verifica se todos os arquivos estão nos lugares corretos
"""

import sys
from pathlib import Path

print("\n" + "="*60)
print("🔍 VERIFICANDO INSTALAÇÃO RLHF")
print("="*60 + "\n")

errors = []
warnings = []

# 1. Verificar estrutura de diretórios
print("📁 Verificando estrutura de diretórios...")

required_dirs = [
    "src/feedback",
    "data/feedback",
]

for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ❌ {dir_path} - FALTANDO!")
        errors.append(f"Diretório {dir_path} não encontrado")

# 2. Verificar arquivos Python
print("\n🐍 Verificando arquivos Python...")

required_files = [
    "src/feedback/__init__.py",
    "src/feedback/feedback_collector.py",
    "src/feedback/feedback_analyzer.py",
    "main_with_feedback.py",
    "train_dpo.py",
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} - FALTANDO!")
        errors.append(f"Arquivo {file_path} não encontrado")

# 3. Verificar documentação
print("\n📚 Verificando documentação...")

doc_files = [
    "RLHF_GUIDE.md",
    "IMPLEMENTACAO_RLHF_README.md",
    "INSTALACAO_ARQUIVOS.md",
]

for file_path in doc_files:
    if Path(file_path).exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ⚠️  {file_path} - opcional, mas recomendado")
        warnings.append(f"Documentação {file_path} não encontrada")

# 4. Testar importações
print("\n🧪 Testando importações...")

try:
    from src.feedback.feedback_collector import FeedbackCollector
    print("   ✅ FeedbackCollector")
except ImportError as e:
    print(f"   ❌ FeedbackCollector - {e}")
    errors.append("Não foi possível importar FeedbackCollector")

try:
    from src.feedback.feedback_analyzer import FeedbackAnalyzer
    print("   ✅ FeedbackAnalyzer")
except ImportError as e:
    print(f"   ❌ FeedbackAnalyzer - {e}")
    errors.append("Não foi possível importar FeedbackAnalyzer")

# 5. Verificar dependências opcionais (DPO)
print("\n📦 Verificando dependências opcionais (DPO)...")

optional_packages = {
    'trl': 'TRL (para DPO training)',
    'peft': 'PEFT (para LoRA)',
    'datasets': 'Datasets (para DPO)',
}

for package, description in optional_packages.items():
    try:
        __import__(package)
        print(f"   ✅ {description}")
    except ImportError:
        print(f"   ⚠️  {description} - Instalar quando for treinar DPO")
        warnings.append(f"{description} não instalado (opcional)")

# Resumo
print("\n" + "="*60)
print("📊 RESUMO DA VERIFICAÇÃO")
print("="*60)

if not errors:
    print("\n✅ INSTALAÇÃO OK!")
    print("\n📋 Próximos passos:")
    print("   1. python main_with_feedback.py")
    print("   2. Começar a coletar feedback")
    print("   3. python src/feedback/feedback_analyzer.py")
else:
    print(f"\n❌ {len(errors)} ERRO(S) ENCONTRADO(S):")
    for error in errors:
        print(f"   - {error}")

if warnings:
    print(f"\n⚠️  {len(warnings)} AVISO(S):")
    for warning in warnings:
        print(f"   - {warning}")

print("\n" + "="*60 + "\n")

sys.exit(0 if not errors else 1)
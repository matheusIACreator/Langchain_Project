"""
Script de verificação da instalação
Testa se todas as dependências críticas foram instaladas corretamente
"""

import sys

print("="*60)
print("🔍 VERIFICANDO INSTALAÇÃO DO PROJETO")
print("="*60)
print(f"\nPython: {sys.version}")
print(f"Versão: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

if sys.version_info.major == 3 and sys.version_info.minor >= 13:
    print("⚠️  AVISO: Python 3.13+ pode ter problemas de compatibilidade")
    print("   Recomendamos Python 3.11 ou 3.12 para melhor estabilidade")

print("\n" + "-"*60)
print("Testando bibliotecas críticas...")
print("-"*60)

# Lista de bibliotecas para testar
tests = []

# PyTorch
try:
    import torch
    cuda_available = torch.cuda.is_available()
    tests.append(("PyTorch", True, torch.__version__))
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA disponível: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA versão: {torch.version.cuda}")
except ImportError as e:
    tests.append(("PyTorch", False, str(e)))
    print(f"❌ PyTorch: {e}")

# LangChain
try:
    import langchain
    tests.append(("LangChain", True, langchain.__version__))
    print(f"✅ LangChain: {langchain.__version__}")
except ImportError as e:
    tests.append(("LangChain", False, str(e)))
    print(f"❌ LangChain: {e}")

# LangChain Community
try:
    import langchain_community
    tests.append(("LangChain Community", True, "OK"))
    print(f"✅ LangChain Community: OK")
except ImportError as e:
    tests.append(("LangChain Community", False, str(e)))
    print(f"❌ LangChain Community: {e}")

# LangChain Hugging Face
try:
    import langchain_huggingface
    tests.append(("LangChain Hugging Face", True, "OK"))
    print(f"✅ LangChain Hugging Face: OK")
except ImportError as e:
    tests.append(("LangChain Hugging Face", False, str(e)))
    print(f"❌ LangChain Hugging Face: {e}")

# ChromaDB
try:
    import chromadb
    tests.append(("ChromaDB", True, chromadb.__version__))
    print(f"✅ ChromaDB: {chromadb.__version__}")
except ImportError as e:
    tests.append(("ChromaDB", False, str(e)))
    print(f"❌ ChromaDB: {e}")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    tests.append(("Sentence Transformers", True, "OK"))
    print(f"✅ Sentence Transformers: OK")
except ImportError as e:
    tests.append(("Sentence Transformers", False, str(e)))
    print(f"❌ Sentence Transformers: {e}")

# Transformers (Hugging Face)
try:
    import transformers
    tests.append(("Transformers", True, transformers.__version__))
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    tests.append(("Transformers", False, str(e)))
    print(f"❌ Transformers: {e}")

# PyPDF
try:
    import pypdf
    tests.append(("PyPDF", True, "OK"))
    print(f"✅ PyPDF: OK")
except ImportError as e:
    tests.append(("PyPDF", False, str(e)))
    print(f"❌ PyPDF: {e}")

# PDFPlumber
try:
    import pdfplumber
    tests.append(("PDFPlumber", True, "OK"))
    print(f"✅ PDFPlumber: OK")
except ImportError as e:
    tests.append(("PDFPlumber", False, str(e)))
    print(f"❌ PDFPlumber: {e}")

# Python-dotenv
try:
    from dotenv import load_dotenv
    tests.append(("Python-dotenv", True, "OK"))
    print(f"✅ Python-dotenv: OK")
except ImportError as e:
    tests.append(("Python-dotenv", False, str(e)))
    print(f"❌ Python-dotenv: {e}")

# Pydantic
try:
    import pydantic
    tests.append(("Pydantic", True, pydantic.__version__))
    print(f"✅ Pydantic: {pydantic.__version__}")
except ImportError as e:
    tests.append(("Pydantic", False, str(e)))
    print(f"❌ Pydantic: {e}")

# NumPy
try:
    import numpy
    tests.append(("NumPy", True, numpy.__version__))
    print(f"✅ NumPy: {numpy.__version__}")
except ImportError as e:
    tests.append(("NumPy", False, str(e)))
    print(f"❌ NumPy: {e}")

# Pandas
try:
    import pandas
    tests.append(("Pandas", True, pandas.__version__))
    print(f"✅ Pandas: {pandas.__version__}")
except ImportError as e:
    tests.append(("Pandas", False, str(e)))
    print(f"❌ Pandas: {e}")

# Resumo
print("\n" + "="*60)
print("RESUMO DA VERIFICAÇÃO")
print("="*60)

passed = sum(1 for _, status, _ in tests if status)
failed = len(tests) - passed

print(f"\n✅ Bibliotecas instaladas: {passed}/{len(tests)}")
if failed > 0:
    print(f"❌ Bibliotecas faltando: {failed}/{len(tests)}")
    print("\n⚠️  Bibliotecas com problemas:")
    for name, status, info in tests:
        if not status:
            print(f"   - {name}")

print("\n" + "="*60)

if failed == 0:
    print("🎉 TUDO PRONTO! Pode começar a usar o projeto!")
    print("\nPróximos passos:")
    print("1. Configure o .env com seu HF_TOKEN")
    print("2. Execute: python src/document_loader.py")
    print("3. Execute: python src/vectorstore.py")
else:
    print("⚠️  ATENÇÃO: Algumas bibliotecas não foram instaladas corretamente")
    print("\nSoluções:")
    print("1. Tente reinstalar com: pip install -r requirements.txt")
    print("2. Veja INSTALL_GUIDE.md para instruções detalhadas")
    print("3. Considere usar Python 3.11 ou 3.12 em vez de 3.13")

print("="*60)

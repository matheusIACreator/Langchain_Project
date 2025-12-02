"""
Setup Script - Configuração rápida do projeto
Executa todos os passos necessários para preparar o sistema
"""

import sys
import subprocess
from pathlib import Path
import os


def print_header(text):
    """Imprime um cabeçalho formatado"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def print_step(step_num, total_steps, description):
    """Imprime o progresso do passo atual"""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-"*60)


def check_env_file():
    """Verifica se o arquivo .env existe"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("⚠️  Arquivo .env não encontrado!")
        print("\n📋 Criando .env a partir do template...")
        
        # Copiar .env.example para .env
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_path)
            print("✅ Arquivo .env criado!")
            print("\n⚠️  IMPORTANTE: Edite o arquivo .env e adicione seu HF_TOKEN")
            print("   Token Hugging Face: https://huggingface.co/settings/tokens")
            
            response = input("\nPressione ENTER quando tiver configurado o .env (ou 's' para sair): ")
            if response.lower() == 's':
                print("Saindo...")
                sys.exit(0)
        else:
            print("❌ Arquivo .env.example não encontrado!")
            sys.exit(1)
    else:
        print("✅ Arquivo .env encontrado!")
        
        # Verificar se o token está configurado
        with open(env_path, 'r') as f:
            content = f.read()
            if "your_huggingface_token_here" in content or "HF_TOKEN=" not in content:
                print("⚠️  HF_TOKEN parece não estar configurado no .env")
                print("   Por favor, edite o arquivo .env e adicione seu token")
                
                response = input("\nToken já configurado? (s/n): ")
                if response.lower() != 's':
                    print("Configure o token e execute novamente.")
                    sys.exit(0)


def check_pdf():
    """Verifica se o PDF existe"""
    pdf_dir = Path("data/raw")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  Nenhum PDF encontrado em data/raw/")
        print("   Por favor, coloque o PDF sobre Galileu na pasta data/raw/")
        sys.exit(1)
    else:
        print(f"✅ PDF encontrado: {pdf_files[0].name}")


def run_command(command, description):
    """Executa um comando Python"""
    try:
        print(f"\n🔄 {description}...")
        result = subprocess.run(
            [sys.executable, command],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(f"✅ {description} concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar {description}")
        print(f"Erro: {e.stderr}")
        return False


def main():
    """Função principal do setup"""
    print_header("🚀 SETUP AUTOMÁTICO - GALILEU GALILEI CHATBOT")
    
    print("Este script irá:")
    print("  1. Verificar configurações")
    print("  2. Processar o documento PDF")
    print("  3. Criar o vector store")
    print("  4. Testar o sistema")
    print()
    
    response = input("Deseja continuar? (s/n): ")
    if response.lower() != 's':
        print("Setup cancelado.")
        return
    
    # Passo 1: Verificar .env
    print_step(1, 4, "Verificando configurações")
    check_env_file()
    
    # Passo 2: Verificar PDF
    print_step(2, 4, "Verificando documento PDF")
    check_pdf()
    
    # Passo 3: Processar documento
    print_step(3, 4, "Processando documento e criando chunks")
    success = run_command("src/document_loader.py", "Processamento do documento")
    if not success:
        print("\n❌ Falha no processamento do documento!")
        print("Execute manualmente: python src/document_loader.py")
        sys.exit(1)
    
    # Passo 4: Criar vector store
    print_step(4, 4, "Criando vector store com embeddings")
    print("⏳ Isso pode demorar alguns minutos (download de modelos)...")
    success = run_command("src/vectorstore.py", "Criação do vector store")
    if not success:
        print("\n❌ Falha na criação do vector store!")
        print("Execute manualmente: python src/vectorstore.py")
        sys.exit(1)
    
    # Conclusão
    print_header("✅ SETUP CONCLUÍDO COM SUCESSO!")
    
    print("🎉 Sistema pronto para uso!\n")
    print("📋 Próximos passos:")
    print("   1. Execute: python main.py")
    print("   2. Acesse a interface web no navegador")
    print("   3. Comece a fazer perguntas sobre Galileu!\n")
    
    print("💡 Dicas:")
    print("   - A primeira execução pode demorar (carregamento do modelo)")
    print("   - Use perguntas específicas para melhores resultados")
    print("   - O sistema mantém contexto da conversa\n")
    
    response = input("Deseja iniciar o chatbot agora? (s/n): ")
    if response.lower() == 's':
        print("\n🚀 Iniciando chatbot...")
        subprocess.run([sys.executable, "main.py"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro durante o setup: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 🔭 RAG System - Galileu Galilei

Sistema de Recuperação e Geração Aumentada (RAG) especializado em Galileu Galilei, o pai da ciência moderna.

## 📋 Sobre o Projeto

Este projeto implementa um chatbot inteligente que responde perguntas sobre a vida, descobertas e contribuições de Galileu Galilei, utilizando:

- **LLM**: Meta Llama-3.1-8B-Instruct (via Hugging Face)
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Framework**: LangChain
- **Memória**: Conversacional com histórico

## 🚀 Funcionalidades

- ✅ Busca semântica em documentos sobre Galileu
- ✅ Respostas contextualizadas baseadas no PDF fornecido
- ✅ Memória de conversação
- ✅ Detecção automática de GPU/CPU
- ✅ Tools opcionais (Wikipedia, Calculator)

## 🛠️ Requisitos

- Python 3.9+
- GPU NVIDIA (opcional, mas recomendado)
- Token da Hugging Face

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/matheusIACreator/Langchain_Project.git
cd Langchain_Project
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

**Para GPU (NVIDIA CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Configure as variáveis de ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o .env e adicione seu token da Hugging Face
# HF_TOKEN=seu_token_aqui
```

Obtenha seu token em: https://huggingface.co/settings/tokens

### 5. Adicione o PDF

Coloque o PDF sobre Galileu Galilei em:
```
data/raw/Galileu_Galilei__O_Pai_da_Ciência_Moderna.pdf
```

## 🎯 Uso

### Processar o documento e criar o vector store

```bash
python src/document_loader.py
```

### Executar o chatbot

```bash
python main.py
```

## 📁 Estrutura do Projeto

```
Langchain_Project/
├── .env                    # Variáveis de ambiente (não versionado)
├── .env.example            # Template de variáveis
├── .gitignore             # Arquivos ignorados pelo Git
├── README.md              # Este arquivo
├── requirements.txt       # Dependências Python
├── main.py               # Ponto de entrada da aplicação
├── config/
│   └── settings.py       # Configurações centralizadas
├── data/
│   ├── raw/              # PDF original
│   ├── processed/        # Documentos processados
│   └── vectorstore/      # Banco de dados ChromaDB
├── src/
│   ├── document_loader.py    # Carrega e processa PDF
│   ├── vectorstore.py        # Gerencia ChromaDB
│   ├── chains/
│   │   └── rag_chain.py      # Chain principal do RAG
│   ├── agents/
│   │   └── rag_agent.py      # Agente com tools
│   ├── tools/
│   │   ├── document_search.py
│   │   └── wikipedia_tool.py
│   ├── prompts/
│   │   └── rag_prompts.py
│   ├── memory/
│   │   └── conversation_memory.py
│   └── utils/
│       └── helpers.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vectorstore_setup.ipynb
│   └── 03_rag_testing.ipynb
└── tests/
```

## 🔧 Configuração de GPU/CPU

O sistema detecta automaticamente se há GPU disponível:

- **GPU NVIDIA (CUDA)**: Utilizado automaticamente se disponível
- **Apple Silicon (MPS)**: Suporte para Mac M1/M2
- **CPU**: Fallback automático se não houver GPU

Para forçar um device específico, edite o `.env`:
```bash
DEVICE=cuda  # ou 'cpu', 'mps'
```

## 📊 Notebooks de Exploração

Execute os notebooks na pasta `notebooks/` para:

1. **01_data_exploration**: Explorar e analisar o PDF
2. **02_vectorstore_setup**: Configurar e testar o ChromaDB
3. **03_rag_testing**: Testar diferentes estratégias de RAG

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

- Reportar bugs
- Sugerir novas funcionalidades
- Enviar pull requests

## 📝 Licença

Este projeto é de uso educacional.

## 👨‍💻 Autor

**Matheus Masago**
- GitHub: [@matheusIACreator](https://github.com/matheusIACreator)
- LinkedIn: [Matheus Masago](https://www.linkedin.com/in/matheus-masago/)

## 🙏 Agradecimentos

- [LangChain](https://www.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [ChromaDB](https://www.trychroma.com/)
- Meta AI (Llama)
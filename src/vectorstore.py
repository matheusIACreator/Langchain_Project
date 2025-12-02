"""
Vector Store Manager - Gerencia ChromaDB e embeddings
Cria, atualiza e consulta o vector store com os documentos processados
"""

import sys
from pathlib import Path
from typing import List, Optional

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import (
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DEVICE,
    DEBUG
)


class GalileuVectorStore:
    """
    Classe para gerenciar o vector store do projeto
    """
    
    def __init__(self):
        """
        Inicializa o vector store e o modelo de embeddings
        """
        print(f"🔧 Inicializando Vector Store...")
        print(f"   Modelo de Embeddings: {EMBEDDING_MODEL}")
        print(f"   Device: {DEVICE}")
        print(f"   Diretório: {CHROMA_PERSIST_DIRECTORY}")
        print(f"   Collection: {COLLECTION_NAME}")
        
        # Configurar embeddings
        self.embeddings = self._setup_embeddings()
        
        # Vector store será inicializado quando necessário
        self.vectorstore = None
    
    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Configura o modelo de embeddings da Hugging Face
        
        Returns:
            Modelo de embeddings configurado
        """
        print(f"\n📥 Baixando/carregando modelo de embeddings...")
        
        try:
            # Configuração do modelo de embeddings
            model_kwargs = {
                'device': DEVICE
            }
            
            encode_kwargs = {
                'normalize_embeddings': True  # Normaliza os embeddings para melhor similaridade
            }
            
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            print(f"✅ Modelo de embeddings carregado com sucesso!")
            
            # Teste rápido
            if DEBUG:
                test_text = "Galileu Galilei foi um cientista italiano."
                test_embedding = embeddings.embed_query(test_text)
                print(f"\n🧪 Teste de embedding:")
                print(f"   Texto: '{test_text}'")
                print(f"   Dimensão do vetor: {len(test_embedding)}")
                print(f"   Primeiros 5 valores: {test_embedding[:5]}")
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"❌ Erro ao configurar embeddings: {str(e)}")
    
    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Cria um novo vector store a partir dos chunks
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            Vector store criado
        """
        print(f"\n🏗️  Criando vector store...")
        print(f"   Total de chunks: {len(chunks)}")
        
        try:
            # Criar o vector store com os chunks
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
            
            # Persistir o vector store
            print(f"💾 Persistindo vector store em disco...")
            # self.vectorstore.persist()  # Não é mais necessário no Chroma v0.4+
            
            print(f"✅ Vector store criado e persistido com sucesso!")
            print(f"   📦 Collection: {COLLECTION_NAME}")
            print(f"   📁 Localização: {CHROMA_PERSIST_DIRECTORY}")
            
            # Estatísticas
            collection_count = self.vectorstore._collection.count()
            print(f"\n📊 Estatísticas do Vector Store:")
            print(f"   Total de embeddings: {collection_count}")
            
            return self.vectorstore
            
        except Exception as e:
            raise Exception(f"❌ Erro ao criar vector store: {str(e)}")
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Carrega um vector store existente
        
        Returns:
            Vector store carregado ou None se não existir
        """
        print(f"\n📂 Tentando carregar vector store existente...")
        
        try:
            # Verificar se o diretório existe e tem arquivos
            persist_path = Path(CHROMA_PERSIST_DIRECTORY)
            if not persist_path.exists() or not any(persist_path.iterdir()):
                print(f"⚠️  Nenhum vector store encontrado em {CHROMA_PERSIST_DIRECTORY}")
                return None
            
            # Carregar o vector store
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
            
            # Verificar se tem dados
            collection_count = self.vectorstore._collection.count()
            
            if collection_count == 0:
                print(f"⚠️  Vector store existe mas está vazio")
                return None
            
            print(f"✅ Vector store carregado com sucesso!")
            print(f"   📦 Collection: {COLLECTION_NAME}")
            print(f"   📊 Total de embeddings: {collection_count}")
            
            return self.vectorstore
            
        except Exception as e:
            print(f"⚠️  Erro ao carregar vector store: {str(e)}")
            return None
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Busca documentos similares à query
        
        Args:
            query: Texto da pergunta/busca
            k: Número de documentos a retornar
            
        Returns:
            Lista de documentos mais relevantes
        """
        if self.vectorstore is None:
            raise ValueError("❌ Vector store não foi inicializado. Execute load_vectorstore() ou create_vectorstore() primeiro.")
        
        print(f"\n🔍 Buscando documentos relevantes...")
        print(f"   Query: '{query}'")
        print(f"   Top K: {k}")
        
        try:
            # Busca por similaridade
            results = self.vectorstore.similarity_search(query, k=k)
            
            print(f"✅ Busca concluída!")
            print(f"   📄 Documentos encontrados: {len(results)}")
            
            if DEBUG:
                print(f"\n{'='*60}")
                print("RESULTADOS DA BUSCA:")
                print(f"{'='*60}")
                for i, doc in enumerate(results, 1):
                    print(f"\n--- Resultado {i} ---")
                    print(f"Página: {doc.metadata.get('page', 'N/A')}")
                    print(f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
                    print(f"Preview: {doc.page_content[:200]}...")
                print(f"{'='*60}\n")
            
            return results
            
        except Exception as e:
            raise Exception(f"❌ Erro ao buscar documentos: {str(e)}")
    
    def search_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """
        Busca documentos com scores de similaridade
        
        Args:
            query: Texto da pergunta/busca
            k: Número de documentos a retornar
            
        Returns:
            Lista de tuplas (documento, score)
        """
        if self.vectorstore is None:
            raise ValueError("❌ Vector store não foi inicializado.")
        
        print(f"\n🔍 Buscando documentos com scores...")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            print(f"✅ Busca concluída!")
            print(f"   📄 Documentos encontrados: {len(results)}")
            
            if DEBUG:
                print(f"\n{'='*60}")
                print("RESULTADOS COM SCORES:")
                print(f"{'='*60}")
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\n--- Resultado {i} ---")
                    print(f"Score: {score:.4f}")
                    print(f"Página: {doc.metadata.get('page', 'N/A')}")
                    print(f"Preview: {doc.page_content[:150]}...")
                print(f"{'='*60}\n")
            
            return results
            
        except Exception as e:
            raise Exception(f"❌ Erro ao buscar documentos: {str(e)}")
    
    def delete_collection(self) -> None:
        """
        Deleta a collection atual (útil para recriar o vector store)
        """
        print(f"\n🗑️  Deletando collection '{COLLECTION_NAME}'...")
        
        try:
            if self.vectorstore is not None:
                self.vectorstore.delete_collection()
                self.vectorstore = None
                print(f"✅ Collection deletada com sucesso!")
            else:
                print(f"⚠️  Nenhuma collection ativa para deletar")
                
        except Exception as e:
            print(f"❌ Erro ao deletar collection: {str(e)}")


def main():
    """
    Função principal para teste standalone
    """
    from document_loader import GalileuDocumentLoader
    
    print("\n" + "="*60)
    print("🚀 CRIANDO VECTOR STORE DO GALILEU")
    print("="*60 + "\n")
    
    try:
        # 1. Carregar e processar documento
        print("📖 Passo 1: Processando documento...")
        loader = GalileuDocumentLoader()
        chunks = loader.process(save_chunks=True)
        
        # 2. Inicializar vector store manager
        print(f"\n🔧 Passo 2: Inicializando Vector Store Manager...")
        vs_manager = GalileuVectorStore()
        
        # 3. Criar vector store
        print(f"\n🏗️  Passo 3: Criando Vector Store...")
        vectorstore = vs_manager.create_vectorstore(chunks)
        
        # 4. Teste de busca
        print(f"\n🧪 Passo 4: Testando busca...")
        test_queries = [
            "Quando Galileu nasceu?",
            "Quais foram as descobertas de Galileu com o telescópio?",
            "O que aconteceu com Galileu e a Igreja?"
        ]
        
        for query in test_queries:
            print(f"\n{'─'*60}")
            results = vs_manager.search(query, k=2)
            print(f"Query: '{query}'")
            print(f"Melhor resultado: {results[0].page_content[:200]}...")
        
        print("\n" + "="*60)
        print("✅ VECTOR STORE CRIADO COM SUCESSO!")
        print("="*60)
        print(f"\n📍 Próximo passo: Execute 'python main.py' para usar o chatbot!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

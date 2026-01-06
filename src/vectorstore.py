"""
Vector Store Manager - Gerencia ChromaDB e embeddings
Versão 2.0: Suporta single-collection (Galileu) e multi-collection (múltiplas figuras)
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

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

# Diretório base para multi-collection
VECTORSTORE_DIR = Path(CHROMA_PERSIST_DIRECTORY).parent / "vectorstore"


class BaseVectorStore:
    """
    Classe base com funcionalidades compartilhadas de embeddings
    """
    
    def __init__(self):
        """
        Inicializa embeddings (compartilhado entre todas as classes)
        """
        self.embeddings = None
    
    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Configura o modelo de embeddings da Hugging Face
        
        Returns:
            Modelo de embeddings configurado
        """
        print(f"\n📥 Carregando modelo de embeddings...")
        
        try:
            # Configuração do modelo de embeddings
            model_kwargs = {'device': DEVICE}
            encode_kwargs = {'normalize_embeddings': True}
            
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            print(f"✅ Modelo de embeddings carregado com sucesso!")
            
            # Teste rápido
            if DEBUG:
                test_text = "Test embedding"
                test_embedding = embeddings.embed_query(test_text)
                print(f"   Dimensão do vetor: {len(test_embedding)}")
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"❌ Erro ao configurar embeddings: {str(e)}")


class GalileuVectorStore(BaseVectorStore):
    """
    Vector Store original para Galileu (v1.0 - backwards compatible)
    """
    
    def __init__(self):
        """
        Inicializa o vector store do Galileu
        """
        super().__init__()
        
        print(f"🔧 Inicializando Vector Store (v1.0 - Galileu)...")
        print(f"   Modelo de Embeddings: {EMBEDDING_MODEL}")
        print(f"   Device: {DEVICE}")
        print(f"   Diretório: {CHROMA_PERSIST_DIRECTORY}")
        print(f"   Collection: {COLLECTION_NAME}")
        
        # Configurar embeddings
        self.embeddings = self._setup_embeddings()
        
        # Vector store será inicializado quando necessário
        self.vectorstore = None
    
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
            raise ValueError("❌ Vector store não foi inicializado.")
        
        if DEBUG:
            print(f"\n🔍 Buscando: '{query}' (top {k})")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            
            if DEBUG:
                print(f"✅ {len(results)} documentos encontrados")
            
            return results
            
        except Exception as e:
            raise Exception(f"❌ Erro ao buscar documentos: {str(e)}")
    
    def search_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """Busca documentos com scores de similaridade"""
        if self.vectorstore is None:
            raise ValueError("❌ Vector store não foi inicializado.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            raise Exception(f"❌ Erro ao buscar documentos: {str(e)}")
    
    def delete_collection(self) -> None:
        """Deleta a collection atual"""
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


class MultiCollectionVectorStore(BaseVectorStore):
    """
    Vector Store v2.0 - Gerencia múltiplas collections
    Organiza por período e figura: renaissance/galileo_galilei, etc.
    """
    
    def __init__(self):
        """
        Inicializa o multi-collection vector store
        """
        super().__init__()
        
        print("\n" + "="*60)
        print("🏗️  INICIALIZANDO MULTI-COLLECTION VECTOR STORE (v2.0)")
        print("="*60 + "\n")
        
        print(f"📁 Diretório base: {VECTORSTORE_DIR}")
        print(f"🔧 Embedding model: {EMBEDDING_MODEL}")
        print(f"⚙️  Device: {DEVICE}")
        
        # Inicializar embeddings (compartilhado)
        self.embeddings = self._setup_embeddings()
        
        # Dicionário de collections: {collection_name: Chroma}
        self.collections = {}
        
        # Descobrir collections existentes
        self._discover_collections()
        
        print("\n✅ Multi-Collection Vector Store inicializado!")
    
    def _discover_collections(self):
        """Descobre collections existentes no diretório"""
        print(f"\n🔍 Descobrindo collections existentes...")
        
        if not VECTORSTORE_DIR.exists():
            print("   ℹ️  Diretório vazio - nenhuma collection encontrada")
            return
        
        # Buscar estrutura: period/figure/
        for period_dir in VECTORSTORE_DIR.iterdir():
            if period_dir.is_dir():
                for figure_dir in period_dir.iterdir():
                    if figure_dir.is_dir() and any(figure_dir.iterdir()):
                        collection_name = f"{period_dir.name}/{figure_dir.name}"
                        print(f"   📦 Encontrada: {collection_name}")
                        self.collections[collection_name] = None  # Lazy loading
        
        if self.collections:
            print(f"✅ {len(self.collections)} collections encontradas")
    
    def create_collection(
        self,
        period: str,
        figure: str,
        chunks: List[Document]
    ) -> Chroma:
        """
        Cria uma nova collection
        
        Args:
            period: Período histórico (ex: "renaissance")
            figure: Nome da figura (ex: "galileo_galilei")
            chunks: Documentos a inserir
            
        Returns:
            Collection criada
        """
        collection_name = f"{period}/{figure}"
        
        print(f"\n🏗️  Criando collection: {collection_name}")
        print(f"   📦 Chunks: {len(chunks)}")
        
        # Diretório da collection
        persist_dir = str(VECTORSTORE_DIR / period / figure)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Nome interno (sem /)
        internal_name = f"{period}_{figure}"
        
        try:
            # Criar vectorstore
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=internal_name,
                persist_directory=persist_dir
            )
            
            # Armazenar
            self.collections[collection_name] = vectorstore
            
            # Verificar
            count = vectorstore._collection.count()
            print(f"✅ Collection criada! ({count} embeddings)")
            
            return vectorstore
            
        except Exception as e:
            print(f"❌ Erro ao criar collection: {str(e)}")
            raise
    
    def load_collection(self, period: str, figure: str) -> Optional[Chroma]:
        """Carrega uma collection existente (lazy loading)"""
        collection_name = f"{period}/{figure}"
        
        # Se já está carregada
        if collection_name in self.collections and self.collections[collection_name]:
            return self.collections[collection_name]
        
        # Verificar se existe
        persist_dir = str(VECTORSTORE_DIR / period / figure)
        if not Path(persist_dir).exists():
            return None
        
        print(f"📂 Carregando: {collection_name}")
        
        try:
            internal_name = f"{period}_{figure}"
            
            vectorstore = Chroma(
                collection_name=internal_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
            
            count = vectorstore._collection.count()
            if count == 0:
                return None
            
            print(f"✅ Carregada! ({count} embeddings)")
            
            self.collections[collection_name] = vectorstore
            return vectorstore
            
        except Exception as e:
            print(f"❌ Erro ao carregar: {str(e)}")
            return None
    
    def search_in_collection(
        self,
        period: str,
        figure: str,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """Busca em uma collection específica"""
        collection_name = f"{period}/{figure}"
        
        # Carregar se necessário
        vectorstore = self.load_collection(period, figure)
        
        if not vectorstore:
            if DEBUG:
                print(f"⚠️  Collection não disponível: {collection_name}")
            return []
        
        if DEBUG:
            print(f"🔍 Buscando em {collection_name}")
        
        results = vectorstore.similarity_search(query, k=k)
        return results
    
    def search_in_multiple(
        self,
        collections: List[Tuple[str, str]],
        query: str,
        k_per_collection: int = 3
    ) -> Dict[str, List[Document]]:
        """Busca em múltiplas collections"""
        if DEBUG:
            print(f"\n🔍 Busca multi-collection: '{query}'")
        
        results = {}
        
        for period, figure in collections:
            docs = self.search_in_collection(period, figure, query, k=k_per_collection)
            
            if docs:
                collection_name = f"{period}/{figure}"
                results[collection_name] = docs
        
        return results
    
    def list_collections(self) -> List[str]:
        """Lista todas as collections disponíveis"""
        self._discover_collections()
        return list(self.collections.keys())
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do sistema"""
        return {
            'total_collections': len(self.collections),
            'collections_loaded': sum(1 for v in self.collections.values() if v),
            'collections_list': list(self.collections.keys()),
            'embedding_model': EMBEDDING_MODEL,
            'device': DEVICE,
        }
    
    def print_stats(self):
        """Imprime estatísticas formatadas"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS DO VECTOR STORE")
        print("="*60)
        print(f"Total de collections: {stats['total_collections']}")
        print(f"Collections carregadas: {stats['collections_loaded']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Device: {stats['device']}")
        
        if stats['collections_list']:
            print(f"\n📦 Collections disponíveis:")
            for col in sorted(stats['collections_list']):
                status = "✅" if self.collections[col] else "💤"
                print(f"   {status} {col}")
        
        print("="*60 + "\n")


def main():
    """Função principal para teste"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'multi'], default='multi')
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Teste v1.0 (Galileu apenas)
        from document_loader import GalileuDocumentLoader
        
        print("\n🧪 MODO: Single Collection (Galileu)")
        loader = GalileuDocumentLoader()
        chunks = loader.process()
        
        vs = GalileuVectorStore()
        vs.create_vectorstore(chunks)
        
        results = vs.search("Quando Galileu nasceu?", k=2)
        print(f"\n✅ Teste concluído! {len(results)} resultados")
        
    else:
        # Teste v2.0 (Multi-collection)
        from src.ingestion.pipeline import MultiPeriodIngestionPipeline
        
        print("\n🧪 MODO: Multi-Collection")
        
        # Processar docs
        pipeline = MultiPeriodIngestionPipeline()
        all_chunks = pipeline.process_all()
        
        # Criar collections
        vs = MultiCollectionVectorStore()
        
        for period, figures in all_chunks.items():
            for figure, chunks in figures.items():
                vs.create_collection(period, figure, chunks)
        
        # Stats
        vs.print_stats()
        
        # Teste de busca
        print("🔍 Testando busca...")
        results = vs.search_in_collection("renaissance", "galileo_galilei", "Galileu nasceu", k=2)
        print(f"✅ {len(results)} resultados encontrados")


if __name__ == "__main__":
    main()
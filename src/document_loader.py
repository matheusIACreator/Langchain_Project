"""
Document Loader - Processa o PDF sobre Galileu Galilei
Carrega, divide em chunks e prepara para inserção no vector store
"""

import sys
from pathlib import Path
from typing import List

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


from config.settings import (
    RAW_DATA_DIR,
    CHUNKS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEBUG
)


class GalileuDocumentLoader:
    """
    Classe para carregar e processar documentos sobre Galileu Galilei
    """
    
    def __init__(self, pdf_path: str = None):
        """
        Inicializa o loader
        
        Args:
            pdf_path: Caminho para o PDF (opcional, usa o padrão se não fornecido)
        """
        if pdf_path is None:
            # Busca automaticamente o PDF na pasta data/raw
            pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(
                    f"❌ Nenhum PDF encontrado em {RAW_DATA_DIR}\n"
                    f"Por favor, coloque o PDF sobre Galileu na pasta data/raw/"
                )
            self.pdf_path = str(pdf_files[0])
            print(f"📄 PDF encontrado: {Path(self.pdf_path).name}")
        else:
            self.pdf_path = pdf_path
        
        # Configurar o text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,  # Adiciona índice de onde o chunk começa
        )
        
    def load_pdf(self) -> List[Document]:
        """
        Carrega o PDF usando PyPDFLoader
        
        Returns:
            Lista de documentos (um por página)
        """
        print(f"📖 Carregando PDF: {self.pdf_path}")
        
        try:
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            print(f"✅ PDF carregado com sucesso!")
            print(f"   📄 Total de páginas: {len(documents)}")
            
            if DEBUG:
                # Mostra preview da primeira página
                print(f"\n{'='*60}")
                print("PREVIEW DA PRIMEIRA PÁGINA:")
                print(f"{'='*60}")
                print(documents[0].page_content[:500] + "...")
                print(f"{'='*60}\n")
            
            return documents
            
        except Exception as e:
            raise Exception(f"❌ Erro ao carregar PDF: {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide os documentos em chunks menores
        
        Args:
            documents: Lista de documentos do PDF
            
        Returns:
            Lista de chunks
        """
        print(f"✂️  Dividindo documentos em chunks...")
        print(f"   Tamanho do chunk: {CHUNK_SIZE} caracteres")
        print(f"   Overlap: {CHUNK_OVERLAP} caracteres")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            print(f"✅ Documentos divididos com sucesso!")
            print(f"   📦 Total de chunks criados: {len(chunks)}")
            
            # Estatísticas dos chunks
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            
            print(f"\n📊 Estatísticas dos chunks:")
            print(f"   Tamanho médio: {avg_size:.0f} caracteres")
            print(f"   Menor chunk: {min_size} caracteres")
            print(f"   Maior chunk: {max_size} caracteres")
            
            if DEBUG:
                # Mostra preview dos primeiros 3 chunks
                print(f"\n{'='*60}")
                print("PREVIEW DOS PRIMEIROS 3 CHUNKS:")
                print(f"{'='*60}")
                for i, chunk in enumerate(chunks[:3], 1):
                    print(f"\n--- Chunk {i} ---")
                    print(f"Página: {chunk.metadata.get('page', 'N/A')}")
                    print(f"Tamanho: {len(chunk.page_content)} caracteres")
                    print(f"Conteúdo: {chunk.page_content[:200]}...")
                print(f"{'='*60}\n")
            
            return chunks
            
        except Exception as e:
            raise Exception(f"❌ Erro ao dividir documentos: {str(e)}")
    
    def add_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Adiciona metadados úteis aos chunks
        
        Args:
            chunks: Lista de chunks
            
        Returns:
            Chunks com metadados enriquecidos
        """
        print(f"🏷️  Adicionando metadados aos chunks...")
        
        for i, chunk in enumerate(chunks):
            # Adiciona ID único ao chunk
            chunk.metadata["chunk_id"] = i
            
            # Adiciona informações sobre o documento fonte
            chunk.metadata["source_document"] = Path(self.pdf_path).name
            
            # Adiciona tipo de documento
            chunk.metadata["document_type"] = "biografia_cientifica"
            
            # Adiciona assunto
            chunk.metadata["subject"] = "Galileu Galilei"
            
            # Calcula densidade de informação (palavras por caractere)
            words = len(chunk.page_content.split())
            chars = len(chunk.page_content)
            chunk.metadata["word_count"] = words
            chunk.metadata["char_count"] = chars
        
        print(f"✅ Metadados adicionados!")
        
        return chunks
    
    def save_chunks(self, chunks: List[Document]) -> None:
        """
        Salva os chunks processados em arquivo (opcional, para debug)
        
        Args:
            chunks: Lista de chunks processados
        """
        output_file = CHUNKS_DIR / "galileu_chunks.txt"
        
        print(f"💾 Salvando chunks em: {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Total de chunks: {len(chunks)}\n")
                f.write("="*80 + "\n\n")
                
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"CHUNK {i}\n")
                    f.write(f"Página: {chunk.metadata.get('page', 'N/A')}\n")
                    f.write(f"Tamanho: {chunk.metadata.get('char_count')} caracteres\n")
                    f.write(f"Palavras: {chunk.metadata.get('word_count')}\n")
                    f.write("-"*80 + "\n")
                    f.write(chunk.page_content + "\n")
                    f.write("="*80 + "\n\n")
            
            print(f"✅ Chunks salvos com sucesso!")
            
        except Exception as e:
            print(f"⚠️  Aviso: Não foi possível salvar chunks: {str(e)}")
    
    def process(self, save_chunks: bool = False) -> List[Document]:
        """
        Pipeline completo de processamento
        
        Args:
            save_chunks: Se True, salva os chunks em arquivo
            
        Returns:
            Lista de chunks processados e prontos para embedding
        """
        print("\n" + "="*60)
        print("🚀 INICIANDO PROCESSAMENTO DO DOCUMENTO")
        print("="*60 + "\n")
        
        # 1. Carregar PDF
        documents = self.load_pdf()
        
        # 2. Dividir em chunks
        chunks = self.split_documents(documents)
        
        # 3. Adicionar metadados
        chunks = self.add_metadata(chunks)
        
        # 4. Salvar chunks (opcional)
        if save_chunks:
            self.save_chunks(chunks)
        
        print("\n" + "="*60)
        print("✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60 + "\n")
        
        return chunks


def main():
    """
    Função principal para teste standalone
    """
    try:
        # Inicializar loader
        loader = GalileuDocumentLoader()
        
        # Processar documento
        chunks = loader.process(save_chunks=True)
        
        print(f"\n✨ Pronto! {len(chunks)} chunks criados e prontos para embedding.")
        print(f"📍 Próximo passo: Execute 'python src/vectorstore.py' para criar o vector store.")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
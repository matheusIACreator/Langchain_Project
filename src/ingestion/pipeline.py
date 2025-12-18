"""
Ingestion Pipeline - Processa biografias e cria chunks para vector store
Suporta múltiplas figuras e períodos organizados por diretório
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import (
    RAW_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEBUG
)


class MultiPeriodIngestionPipeline:
    """
    Pipeline para ingestão de múltiplas figuras históricas
    """
    
    def __init__(self):
        """
        Inicializa o pipeline
        """
        print("🔧 Inicializando Multi-Period Ingestion Pipeline...")
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
        
        print("✅ Pipeline inicializado!")
    
    def discover_figures(self) -> Dict[str, List[Path]]:
        """
        Descobre automaticamente figuras organizadas por período
        
        Returns:
            Dict mapeando período -> lista de arquivos
        """
        print(f"\n🔍 Descobrindo figuras em {RAW_DATA_DIR}...")
        
        figures_by_period = {}
        
        # Buscar em cada diretório de período
        for period_dir in RAW_DATA_DIR.iterdir():
            if period_dir.is_dir() and period_dir.name in ['renaissance', 'enlightenment', 'modern_era']:
                # Buscar arquivos .txt neste período
                txt_files = list(period_dir.glob("*.txt"))
                
                if txt_files:
                    figures_by_period[period_dir.name] = txt_files
                    print(f"   📁 {period_dir.name}: {len(txt_files)} figuras")
                    for file in txt_files:
                        print(f"      - {file.stem}")
        
        total_figures = sum(len(files) for files in figures_by_period.values())
        print(f"\n✅ Total de figuras encontradas: {total_figures}")
        
        return figures_by_period
    
    def load_biography(self, file_path: Path) -> str:
        """
        Carrega biografia de arquivo
        
        Args:
            file_path: Caminho do arquivo .txt
            
        Returns:
            Texto da biografia
        """
        print(f"\n📖 Carregando: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"   Tamanho: {len(text)} caracteres")
            return text
            
        except Exception as e:
            print(f"❌ Erro ao carregar {file_path.name}: {str(e)}")
            return ""
    
    def process_figure(
        self,
        file_path: Path,
        period: str
    ) -> Tuple[str, str, List[Document]]:
        """
        Processa uma figura: carrega, chunka, adiciona metadata
        
        Args:
            file_path: Caminho do arquivo
            period: Período histórico
            
        Returns:
            Tuple (period, figure_name, chunks)
        """
        # Extrair nome da figura do arquivo
        figure_name = file_path.stem  # Ex: "isaac_newton"
        
        print(f"\n{'='*60}")
        print(f"⚙️  PROCESSANDO: {figure_name} ({period})")
        print(f"{'='*60}")
        
        # Carregar texto
        text = self.load_biography(file_path)
        
        if not text:
            return period, figure_name, []
        
        # Criar documento base
        base_doc = Document(
            page_content=text,
            metadata={
                'source': str(file_path),
                'figure': figure_name,
                'period': period,
            }
        )
        
        # Dividir em chunks
        print(f"✂️  Dividindo em chunks...")
        chunks = self.text_splitter.split_documents([base_doc])
        
        # Enriquecer metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'char_count': len(chunk.page_content),
                'word_count': len(chunk.page_content.split()),
            })
        
        # Estatísticas
        total_chars = sum(len(c.page_content) for c in chunks)
        avg_size = total_chars / len(chunks) if chunks else 0
        
        print(f"✅ Processamento concluído!")
        print(f"   📦 Chunks criados: {len(chunks)}")
        print(f"   📊 Tamanho médio: {avg_size:.0f} caracteres")
        
        if DEBUG and chunks:
            print(f"\n   Preview do primeiro chunk:")
            print(f"   {chunks[0].page_content[:200]}...")
        
        return period, figure_name, chunks
    
    def process_all(self) -> Dict[str, Dict[str, List[Document]]]:
        """
        Processa todas as figuras descobertas
        
        Returns:
            Dict estruturado: {period: {figure: chunks}}
        """
        print("\n" + "="*60)
        print("🚀 INICIANDO PROCESSAMENTO DE TODAS AS FIGURAS")
        print("="*60)
        
        # Descobrir figuras
        figures_by_period = self.discover_figures()
        
        if not figures_by_period:
            print("\n⚠️  Nenhuma figura encontrada para processar!")
            return {}
        
        # Processar cada figura
        all_chunks = {}
        
        for period, files in figures_by_period.items():
            all_chunks[period] = {}
            
            for file_path in files:
                _, figure_name, chunks = self.process_figure(file_path, period)
                
                if chunks:
                    all_chunks[period][figure_name] = chunks
        
        # Resumo final
        print("\n" + "="*60)
        print("✅ PROCESSAMENTO CONCLUÍDO!")
        print("="*60)
        
        for period, figures in all_chunks.items():
            print(f"\n📁 {period}:")
            for figure, chunks in figures.items():
                total_chars = sum(len(c.page_content) for c in chunks)
                print(f"   • {figure}: {len(chunks)} chunks ({total_chars:,} chars)")
        
        return all_chunks


def main():
    """
    Função principal para teste standalone
    """
    try:
        # Inicializar pipeline
        pipeline = MultiPeriodIngestionPipeline()
        
        # Processar todas as figuras
        all_chunks = pipeline.process_all()
        
        # Estatísticas gerais
        total_figures = sum(len(figures) for figures in all_chunks.values())
        total_chunks = sum(
            len(chunks)
            for figures in all_chunks.values()
            for chunks in figures.values()
        )
        
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   Total de períodos: {len(all_chunks)}")
        print(f"   Total de figuras: {total_figures}")
        print(f"   Total de chunks: {total_chunks}")
        
        print(f"\n✨ Pipeline pronto!")
        print(f"📍 Próximo passo: Criar multi-collection vector store")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
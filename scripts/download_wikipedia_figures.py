"""
Wikipedia API Downloader
Usa a API oficial da Wikipedia para baixar biografias
Mais confiável que web scraping
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import requests
import json
import time

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class WikipediaAPIDownloader:
    """
    Baixa biografias usando a API oficial da Wikipedia
    """
    
    def __init__(self, language: str = "en"):
        """
        Inicializa o downloader
        
        Args:
            language: Código do idioma (pt, en, etc)
        """
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Headers educados
        self.headers = {
            'User-Agent': 'HistoricalFiguresBot/2.0 (Educational Project)',
            'Accept': 'application/json'
        }
        
    def download_figure_bio(self, figure_name: str, output_dir: Path) -> bool:
        """
        Baixa biografia usando API
        
        Args:
            figure_name: Nome da figura (ex: "Isaac_Newton")
            output_dir: Diretório de saída
            
        Returns:
            True se sucesso
        """
        print(f"\n📥 Baixando biografia de {figure_name}...")
        
        try:
            # Normalizar nome
            page_title = figure_name.replace("_", " ")
            
            # Parâmetros da API
            params = {
                'action': 'query',
                'format': 'json',
                'titles': page_title,
                'prop': 'extracts|info',
                'explaintext': True,  # Texto puro sem HTML
                'exsectionformat': 'plain',
                'inprop': 'url'
            }
            
            print(f"   Consultando API da Wikipedia...")
            
            # Fazer requisição
            response = requests.get(
                self.api_url,
                params=params,
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extrair página
            pages = data.get('query', {}).get('pages', {})
            
            if not pages:
                print(f"❌ Nenhuma página encontrada")
                return False
            
            # Pegar primeira (e única) página
            page = list(pages.values())[0]
            
            # Verificar se página existe
            if 'missing' in page:
                print(f"❌ Página não existe: {page_title}")
                return False
            
            # Extrair conteúdo
            title = page.get('title', figure_name)
            extract = page.get('extract', '')
            url = page.get('fullurl', '')
            
            if not extract:
                print(f"❌ Conteúdo vazio")
                return False
            
            # Estruturar conteúdo
            content = self._structure_content(title, extract, url)
            
            # Salvar
            output_file = output_dir / f"{figure_name.lower().replace(' ', '_')}.txt"
            self._save_content(content, output_file)
            
            print(f"✅ Biografia salva: {output_file.name}")
            print(f"   Tamanho: {len(extract)} caracteres")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro de rede: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Erro ao processar {figure_name}: {str(e)}")
            return False
    
    def _structure_content(self, title: str, extract: str, url: str) -> Dict:
        """
        Estrutura o conteúdo extraído
        
        Args:
            title: Título da página
            extract: Texto extraído
            url: URL da página
            
        Returns:
            Dict com conteúdo estruturado
        """
        # Dividir em parágrafos
        paragraphs = [p.strip() for p in extract.split('\n') if p.strip()]
        
        # Primeiro parágrafo é geralmente o resumo
        summary = paragraphs[0] if paragraphs else ""
        
        # Tentar identificar seções (básico)
        sections = []
        current_section = {"title": "Main Content", "paragraphs": []}
        
        for para in paragraphs:
            # Heurística simples: parágrafos muito curtos podem ser títulos
            if len(para) < 50 and para.isupper():
                if current_section["paragraphs"]:
                    sections.append(current_section)
                current_section = {"title": para, "paragraphs": []}
            else:
                current_section["paragraphs"].append(para)
        
        if current_section["paragraphs"]:
            sections.append(current_section)
        
        return {
            'title': title,
            'url': url,
            'summary': summary,
            'sections': sections
        }
    
    def _save_content(self, content: Dict, output_file: Path):
        """
        Salva conteúdo em arquivo
        
        Args:
            content: Dict com conteúdo
            output_file: Arquivo de saída
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Cabeçalho
            f.write(f"# {content['title']}\n\n")
            f.write(f"**Fonte**: {content['url']}\n\n")
            f.write("---\n\n")
            
            # Resumo
            f.write("## Resumo\n\n")
            f.write(content['summary'] + "\n\n")
            f.write("---\n\n")
            
            # Seções
            f.write("## Conteúdo Completo\n\n")
            for section in content['sections']:
                f.write(f"### {section['title']}\n\n")
                for para in section['paragraphs']:
                    f.write(para + "\n\n")
    
    def download_multiple_figures(
        self,
        figures: List[str],
        output_base_dir: Path,
        periods: Dict[str, str] = None
    ) -> Dict[str, bool]:
        """
        Baixa múltiplas figuras
        
        Args:
            figures: Lista de nomes de figuras
            output_base_dir: Diretório base
            periods: Mapeamento figura -> período
            
        Returns:
            Dict com status de cada download
        """
        results = {}
        
        for i, figure in enumerate(figures, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(figures)}] {figure}")
            print('='*60)
            
            # Determinar diretório de saída
            if periods and figure in periods:
                period = periods[figure]
                output_dir = output_base_dir / period
            else:
                output_dir = output_base_dir
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Baixar
            success = self.download_figure_bio(figure, output_dir)
            results[figure] = success
            
            # Rate limiting: aguardar entre requisições
            if i < len(figures):
                print("   ⏳ Aguardando 1s...")
                time.sleep(1)
        
        return results


# ===== CONFIGURAÇÕES =====

FIGURES_BY_PERIOD = {
    'renaissance': [
        'Galileo Galilei',
        'Leonardo da Vinci',
        'Michelangelo',
    ],
    'enlightenment': [
        'Isaac Newton',
        'Voltaire',
        'Benjamin Franklin',
    ],
    'modern_era': [
        'Albert Einstein',
        'Marie Curie',
        'Charles Darwin',
        'Nikola Tesla',
    ],
}


def main():
    """
    Função principal
    """
    parser = argparse.ArgumentParser(
        description='Download Wikipedia biographies using API'
    )
    parser.add_argument(
        '--figures',
        type=str,
        help='Comma-separated list (ex: "Galileo Galilei,Isaac Newton")'
    )
    parser.add_argument(
        '--period',
        type=str,
        choices=['renaissance', 'enlightenment', 'modern_era', 'all'],
        default=None,
        help='Download all figures from a period'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        choices=['pt', 'en', 'es', 'fr', 'de'],
        help='Wikipedia language'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📚 WIKIPEDIA API BIOGRAPHY DOWNLOADER")
    print("="*60 + "\n")
    
    # Determinar figuras
    if args.figures:
        figures = [f.strip() for f in args.figures.split(',')]
        periods_map = {}
    elif args.period:
        if args.period == 'all':
            figures = []
            periods_map = {}
            for period, period_figures in FIGURES_BY_PERIOD.items():
                for figure in period_figures:
                    figures.append(figure)
                    periods_map[figure] = period
        else:
            figures = FIGURES_BY_PERIOD[args.period]
            periods_map = {fig: args.period for fig in figures}
    else:
        print("❌ Erro: Especifique --figures ou --period")
        return
    
    print(f"📋 Figuras a baixar ({len(figures)}):")
    for fig in figures:
        period = periods_map.get(fig, 'geral')
        print(f"   • {fig} ({period})")
    print()
    
    # Inicializar downloader
    downloader = WikipediaAPIDownloader(language=args.language)
    
    # Baixar
    output_dir = Path(args.output_dir)
    results = downloader.download_multiple_figures(figures, output_dir, periods_map)
    
    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO")
    print("="*60 + "\n")
    
    successes = sum(1 for success in results.values() if success)
    failures = len(results) - successes
    
    print(f"✅ Sucessos: {successes}/{len(results)}")
    print(f"❌ Falhas: {failures}/{len(results)}")
    
    if failures > 0:
        print("\n⚠️  Falhas:")
        for figure, success in results.items():
            if not success:
                print(f"   • {figure}")
    
    if successes > 0:
        print(f"\n📁 Arquivos salvos em: {output_dir}")
    
    print("\n✨ Concluído!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""
Wikipedia Figure Downloader
Baixa biografias da Wikipedia e converte para formato processável
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import re

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class WikipediaDownloader:
    """
    Baixa e processa biografias da Wikipedia
    """
    
    def __init__(self, language: str = "pt"):
        """
        Inicializa o downloader
        
        Args:
            language: Código do idioma (pt, en, etc)
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/wiki/"
        
    def download_figure_bio(self, figure_name: str, output_dir: Path) -> bool:
        """
        Baixa biografia de uma figura
        
        Args:
            figure_name: Nome da figura (ex: "Isaac_Newton")
            output_dir: Diretório de saída
            
        Returns:
            True se sucesso
        """
        print(f"\n📥 Baixando biografia de {figure_name}...")
        
        try:
            # Construir URL
            url = self.base_url + figure_name.replace(" ", "_")
            print(f"   URL: {url}")
            
            # Fazer request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parsear HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extrair conteúdo
            content = self._extract_content(soup, figure_name)
            
            if not content:
                print(f"❌ Falha ao extrair conteúdo")
                return False
            
            # Salvar como texto
            output_file = output_dir / f"{figure_name.lower().replace(' ', '_')}.txt"
            self._save_content(content, output_file)
            
            print(f"✅ Biografia salva: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao baixar {figure_name}: {str(e)}")
            return False
    
    def _extract_content(self, soup: BeautifulSoup, figure_name: str) -> Dict:
        """
        Extrai conteúdo relevante da página
        
        Args:
            soup: BeautifulSoup object
            figure_name: Nome da figura
            
        Returns:
            Dict com conteúdo estruturado
        """
        content = {
            'title': figure_name,
            'summary': '',
            'sections': []
        }
        
        # Extrair resumo (primeiro parágrafo)
        first_para = soup.find('div', class_='mw-parser-output').find('p')
        if first_para:
            content['summary'] = self._clean_text(first_para.get_text())
        
        # Extrair seções
        current_section = None
        for element in soup.find('div', class_='mw-parser-output').children:
            # Detectar heading (seção)
            if element.name in ['h2', 'h3', 'h4']:
                if current_section:
                    content['sections'].append(current_section)
                
                section_title = self._clean_text(element.get_text())
                current_section = {
                    'title': section_title,
                    'content': []
                }
            
            # Adicionar parágrafo à seção atual
            elif element.name == 'p' and current_section:
                text = self._clean_text(element.get_text())
                if text:
                    current_section['content'].append(text)
        
        # Adicionar última seção
        if current_section:
            content['sections'].append(current_section)
        
        return content
    
    def _clean_text(self, text: str) -> str:
        """
        Limpa texto (remove referências, etc)
        
        Args:
            text: Texto bruto
            
        Returns:
            Texto limpo
        """
        # Remover referências [1], [2], etc
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remover espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Remover "Editar" (botões de edição)
        text = text.replace('[editar]', '')
        text = text.replace('[Editar]', '')
        
        return text.strip()
    
    def _save_content(self, content: Dict, output_file: Path):
        """
        Salva conteúdo em arquivo de texto estruturado
        
        Args:
            content: Dict com conteúdo
            output_file: Arquivo de saída
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Título
            f.write(f"# {content['title']}\n\n")
            
            # Resumo
            f.write("## Resumo\n\n")
            f.write(content['summary'] + "\n\n")
            
            # Seções
            for section in content['sections']:
                # Ignorar seções não relevantes
                if any(skip in section['title'].lower() for skip in 
                       ['referências', 'bibliografia', 'ligações externas', 
                        'ver também', 'notas']):
                    continue
                
                f.write(f"## {section['title']}\n\n")
                for para in section['content']:
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
        
        for figure in figures:
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
        
        return results


# ===== CONFIGURAÇÕES PADRÃO =====

FIGURES_BY_PERIOD = {
    'renaissance': [
        'Galileo_Galilei',
        'Leonardo_da_Vinci',
        'Michelangelo',
    ],
    'enlightenment': [
        'Isaac_Newton',
        'Voltaire',
        'Benjamin_Franklin',
    ],
    'modern_era': [
        'Albert_Einstein',
        'Marie_Curie',
        'Charles_Darwin',
        'Nikola_Tesla',
    ],
}

# Mapeamento para português (se usar Wikipedia PT)
FIGURES_PT = {
    'Galileo_Galilei': 'Galileu_Galilei',
    'Isaac_Newton': 'Isaac_Newton',
    'Albert_Einstein': 'Albert_Einstein',
    'Leonardo_da_Vinci': 'Leonardo_da_Vinci',
    'Marie_Curie': 'Marie_Curie',
    'Charles_Darwin': 'Charles_Darwin',
}


def main():
    """
    Função principal
    """
    parser = argparse.ArgumentParser(description='Download Wikipedia biographies')
    parser.add_argument(
        '--figures',
        type=str,
        help='Comma-separated list of figures (ex: "Galileo_Galilei,Isaac_Newton")'
    )
    parser.add_argument(
        '--period',
        type=str,
        choices=['renaissance', 'enlightenment', 'modern_era', 'all'],
        default='all',
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
        default='pt',
        choices=['pt', 'en'],
        help='Wikipedia language'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📚 WIKIPEDIA BIOGRAPHY DOWNLOADER")
    print("="*60 + "\n")
    
    # Determinar figuras a baixar
    if args.figures:
        figures = [f.strip() for f in args.figures.split(',')]
        periods_map = {}
    elif args.period == 'all':
        figures = []
        periods_map = {}
        for period, period_figures in FIGURES_BY_PERIOD.items():
            for figure in period_figures:
                figures.append(figure)
                periods_map[figure] = period
    else:
        figures = FIGURES_BY_PERIOD[args.period]
        periods_map = {fig: args.period for fig in figures}
    
    # Ajustar para português se necessário
    if args.language == 'pt':
        figures = [FIGURES_PT.get(fig, fig) for fig in figures]
    
    print(f"Figuras a baixar ({len(figures)}):")
    for fig in figures:
        period = periods_map.get(fig, 'unknown')
        print(f"   - {fig} ({period})")
    
    # Inicializar downloader
    downloader = WikipediaDownloader(language=args.language)
    
    # Baixar
    output_dir = Path(args.output_dir)
    results = downloader.download_multiple_figures(figures, output_dir, periods_map)
    
    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO DOS DOWNLOADS")
    print("="*60 + "\n")
    
    successes = sum(1 for success in results.values() if success)
    failures = len(results) - successes
    
    print(f"✅ Sucessos: {successes}")
    print(f"❌ Falhas: {failures}")
    
    if failures > 0:
        print("\nFalhas:")
        for figure, success in results.items():
            if not success:
                print(f"   - {figure}")
    
    print("\n✨ Concluído!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

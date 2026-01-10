"""
Feedback Collector - Sistema de coleta de feedback humano
Armazena avaliações das respostas para análise e melhoria do modelo
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import sqlite3

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Determinar BASE_DIR sem importar config.settings (que importa torch)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class FeedbackCollector:
    """
    Coleta e armazena feedback humano sobre as respostas do chatbot
    """
    
    def __init__(self, db_path: str = None):
        """
        Inicializa o coletor de feedback
        
        Args:
            db_path: Caminho para o banco de dados SQLite
        """
        if db_path is None:
            feedback_dir = BASE_DIR / "data" / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(feedback_dir / "feedback.db")
        
        self.db_path = db_path
        self._init_database()
        
        print(f"📊 Feedback Collector inicializado: {db_path}")
    
    def _init_database(self):
        """
        Inicializa o banco de dados com as tabelas necessárias
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela principal de feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                rating INTEGER,
                thumbs_up BOOLEAN,
                comment TEXT,
                source_documents TEXT,
                metadata TEXT
            )
        """)
        
        # Tabela de pares de preferência (para DPO/RLHF)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                response_chosen TEXT NOT NULL,
                response_rejected TEXT NOT NULL,
                reason TEXT,
                metadata TEXT
            )
        """)
        
        # Índices para busca rápida
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON feedback(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rating 
            ON feedback(rating)
        """)
        
        conn.commit()
        conn.close()
    
    def add_feedback(
        self,
        query: str,
        response: str,
        rating: Optional[int] = None,
        thumbs_up: Optional[bool] = None,
        comment: Optional[str] = None,
        source_documents: Optional[List[Dict]] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Adiciona um feedback ao banco de dados
        
        Args:
            query: Pergunta do usuário
            response: Resposta do chatbot
            rating: Rating de 1-5 (opcional)
            thumbs_up: True/False para thumbs up/down (opcional)
            comment: Comentário adicional do usuário
            source_documents: Documentos usados na resposta
            session_id: ID da sessão
            metadata: Metadados adicionais
            
        Returns:
            ID do feedback inserido
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Serializar documentos e metadata como JSON
        source_docs_json = json.dumps(source_documents) if source_documents else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO feedback 
            (timestamp, session_id, query, response, rating, thumbs_up, comment, 
             source_documents, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            session_id,
            query,
            response,
            rating,
            thumbs_up,
            comment,
            source_docs_json,
            metadata_json
        ))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Feedback registrado (ID: {feedback_id})")
        
        return feedback_id
    
    def add_preference_pair(
        self,
        query: str,
        response_chosen: str,
        response_rejected: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Adiciona um par de preferência (para DPO/RLHF)
        
        Args:
            query: Pergunta
            response_chosen: Resposta escolhida (melhor)
            response_rejected: Resposta rejeitada (pior)
            reason: Razão da escolha
            metadata: Metadados adicionais
            
        Returns:
            ID do par inserido
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO preference_pairs 
            (timestamp, query, response_chosen, response_rejected, reason, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            query,
            response_chosen,
            response_rejected,
            reason,
            metadata_json
        ))
        
        pair_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Par de preferência registrado (ID: {pair_id})")
        
        return pair_id
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre o feedback coletado
        
        Returns:
            Dict com estatísticas
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total de feedbacks
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]
        
        # Feedbacks com rating
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating IS NOT NULL")
        with_rating = cursor.fetchone()[0]
        
        # Média de rating
        cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
        avg_rating = cursor.fetchone()[0] or 0
        
        # Thumbs up vs down
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE thumbs_up = 1")
        thumbs_up = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE thumbs_up = 0")
        thumbs_down = cursor.fetchone()[0]
        
        # Feedbacks com comentário
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE comment IS NOT NULL AND comment != ''")
        with_comment = cursor.fetchone()[0]
        
        # Pares de preferência
        cursor.execute("SELECT COUNT(*) FROM preference_pairs")
        preference_pairs = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_feedbacks": total,
            "with_rating": with_rating,
            "avg_rating": round(avg_rating, 2),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "with_comment": with_comment,
            "preference_pairs": preference_pairs
        }
    
    def get_all_feedbacks(
        self,
        limit: Optional[int] = None,
        min_rating: Optional[int] = None,
        thumbs_up_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Recupera feedbacks do banco de dados
        
        Args:
            limit: Número máximo de feedbacks
            min_rating: Rating mínimo para filtrar
            thumbs_up_only: Retornar apenas thumbs up
            
        Returns:
            Lista de feedbacks
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []
        
        if min_rating is not None:
            query += " AND rating >= ?"
            params.append(min_rating)
        
        if thumbs_up_only:
            query += " AND thumbs_up = 1"
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        feedbacks = []
        for row in cursor.fetchall():
            feedback = dict(zip(columns, row))
            
            # Deserializar JSON
            if feedback['source_documents']:
                feedback['source_documents'] = json.loads(feedback['source_documents'])
            if feedback['metadata']:
                feedback['metadata'] = json.loads(feedback['metadata'])
            
            feedbacks.append(feedback)
        
        conn.close()
        
        return feedbacks
    
    def export_for_training(
        self,
        output_file: str,
        min_rating: int = 4,
        format: str = "jsonl"
    ) -> int:
        """
        Exporta feedbacks positivos para treinamento
        
        Args:
            output_file: Arquivo de saída
            min_rating: Rating mínimo para considerar
            format: Formato do arquivo (jsonl ou json)
            
        Returns:
            Número de exemplos exportados
        """
        feedbacks = self.get_all_feedbacks(min_rating=min_rating)
        
        # Filtrar apenas feedbacks úteis
        training_data = []
        for fb in feedbacks:
            if fb['rating'] and fb['rating'] >= min_rating:
                example = {
                    "query": fb['query'],
                    "response": fb['response'],
                    "rating": fb['rating']
                }
                training_data.append(example)
        
        # Exportar
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {len(training_data)} exemplos exportados para {output_file}")
        
        return len(training_data)
    
    def export_preference_pairs(self, output_file: str) -> int:
        """
        Exporta pares de preferência para treinamento DPO
        
        Args:
            output_file: Arquivo de saída
            
        Returns:
            Número de pares exportados
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM preference_pairs ORDER BY timestamp DESC")
        columns = [desc[0] for desc in cursor.description]
        
        pairs = []
        for row in cursor.fetchall():
            pair = dict(zip(columns, row))
            if pair['metadata']:
                pair['metadata'] = json.loads(pair['metadata'])
            pairs.append(pair)
        
        conn.close()
        
        # Exportar
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {len(pairs)} pares de preferência exportados para {output_file}")
        
        return len(pairs)


def main():
    """
    Teste standalone do sistema de feedback
    """
    print("\n" + "="*60)
    print("🧪 TESTANDO FEEDBACK COLLECTOR")
    print("="*60 + "\n")
    
    # Inicializar coletor
    collector = FeedbackCollector()
    
    # Adicionar alguns feedbacks de exemplo
    print("📝 Adicionando feedbacks de exemplo...\n")
    
    collector.add_feedback(
        query="Quando Galileu nasceu?",
        response="Galileu Galilei nasceu em 15 de fevereiro de 1564, em Pisa, Itália.",
        rating=5,
        thumbs_up=True,
        comment="Resposta perfeita e precisa!"
    )
    
    collector.add_feedback(
        query="Quais foram suas descobertas?",
        response="Galileu fez diversas descobertas importantes...",
        rating=4,
        thumbs_up=True
    )
    
    collector.add_feedback(
        query="O que é física quântica?",
        response="Desculpe, sou especializado em Galileu Galilei...",
        rating=3,
        thumbs_up=False,
        comment="Entendo a limitação, mas seria útil ter mais informações."
    )
    
    # Adicionar par de preferência
    print("\n📊 Adicionando par de preferência...\n")
    
    collector.add_preference_pair(
        query="Quando Galileu morreu?",
        response_chosen="Galileu Galilei faleceu em 8 de janeiro de 1642, em Arcetri, perto de Florença, aos 77 anos de idade.",
        response_rejected="Galileu morreu em 1642.",
        reason="A primeira resposta é mais completa e informativa."
    )
    
    # Mostrar estatísticas
    print("\n📊 Estatísticas:")
    print("-" * 60)
    stats = collector.get_feedback_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Exportar para treinamento
    print("\n💾 Exportando dados...")
    collector.export_for_training("data/feedback/training_data.jsonl", min_rating=4)
    collector.export_preference_pairs("data/feedback/preference_pairs.json")
    
    print("\n✅ Teste concluído com sucesso!")


if __name__ == "__main__":
    main()
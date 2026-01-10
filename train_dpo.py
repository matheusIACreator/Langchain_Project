"""
DPO Training Script - Treina modelo com Direct Preference Optimization
Usa feedback coletado para melhorar o modelo através de pares de preferência
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any
from collections import defaultdict

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

from config.settings import MODEL_NAME, HF_TOKEN
from src.feedback.feedback_collector import FeedbackCollector


def prepare_dpo_dataset(min_samples: int = 50) -> Dataset:
    """
    Prepara dataset DPO a partir do feedback coletado
    
    Args:
        min_samples: Número mínimo de pares necessários
        
    Returns:
        Dataset formatado para DPO
    """
    print("\n📊 Preparando dataset DPO...")
    
    collector = FeedbackCollector()
    feedbacks = collector.get_all_feedbacks()
    
    if len(feedbacks) < min_samples:
        raise ValueError(
            f"❌ Poucos dados! Tem {len(feedbacks)}, precisa de pelo menos {min_samples}\n"
            f"   Continue coletando feedback antes de treinar."
        )
    
    # Agrupar feedbacks por query
    by_query = defaultdict(list)
    
    for fb in feedbacks:
        if fb['rating'] is not None:  # Só usar feedbacks com rating
            by_query[fb['query']].append(fb)
    
    # Criar pares de preferência: melhor resposta vs pior resposta
    dpo_data = []
    
    for query, responses in by_query.items():
        if len(responses) < 2:
            continue
        
        # Ordenar por rating
        responses.sort(key=lambda x: x['rating'], reverse=True)
        
        # Estratégia 1: Melhor vs Pior
        best = responses[0]
        worst = responses[-1]
        
        if best['rating'] > worst['rating']:
            dpo_data.append({
                'prompt': query,
                'chosen': best['response'],
                'rejected': worst['response'],
                'metadata': {
                    'chosen_rating': best['rating'],
                    'rejected_rating': worst['rating']
                }
            })
        
        # Estratégia 2: Se tiver muitas respostas, criar múltiplos pares
        if len(responses) >= 4:
            # Segundo melhor vs segundo pior
            second_best = responses[1]
            second_worst = responses[-2]
            
            if second_best['rating'] > second_worst['rating']:
                dpo_data.append({
                    'prompt': query,
                    'chosen': second_best['response'],
                    'rejected': second_worst['response'],
                    'metadata': {
                        'chosen_rating': second_best['rating'],
                        'rejected_rating': second_worst['rating']
                    }
                })
    
    print(f"✅ {len(dpo_data)} pares de preferência criados")
    
    if len(dpo_data) < min_samples:
        raise ValueError(
            f"❌ Poucos pares! Tem {len(dpo_data)}, precisa de pelo menos {min_samples}\n"
            f"   Precisa de mais diversidade nas queries (diferentes perguntas com múltiplas respostas)."
        )
    
    # Estatísticas
    chosen_ratings = [pair['metadata']['chosen_rating'] for pair in dpo_data]
    rejected_ratings = [pair['metadata']['rejected_rating'] for pair in dpo_data]
    
    print(f"\n📊 Estatísticas do dataset:")
    print(f"   Total de pares: {len(dpo_data)}")
    print(f"   Rating médio (chosen): {sum(chosen_ratings)/len(chosen_ratings):.2f}")
    print(f"   Rating médio (rejected): {sum(rejected_ratings)/len(rejected_ratings):.2f}")
    print(f"   Diferença média: {(sum(chosen_ratings)-sum(rejected_ratings))/len(chosen_ratings):.2f}")
    
    # Converter para Dataset
    dataset = Dataset.from_list(dpo_data)
    
    # Salvar dataset
    output_dir = Path("data/dpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset.save_to_disk(str(output_dir / "dataset"))
    print(f"✅ Dataset salvo em: {output_dir / 'dataset'}")
    
    # Salvar JSON para inspeção
    with open(output_dir / "dataset.json", 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    return dataset


def train_dpo_model(
    dataset: Dataset,
    output_dir: str = "models/galileu_dpo",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5
):
    """
    Treina modelo usando DPO
    
    Args:
        dataset: Dataset preparado
        output_dir: Diretório de saída
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
    """
    print("\n" + "="*60)
    print("🚀 INICIANDO TREINAMENTO DPO")
    print("="*60 + "\n")
    
    # Verificar recursos
    if not torch.cuda.is_available():
        print("⚠️  ATENÇÃO: GPU não detectada. Treinamento será muito lento!")
        print("   Recomendamos usar Google Colab ou serviço com GPU.")
        response = input("\nContinuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            print("Treinamento cancelado.")
            return
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Carregar modelo base
    print(f"\n📥 Carregando modelo base: {MODEL_NAME}")
    print("   Isso pode demorar alguns minutos...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # Configurar padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        print("✅ Modelo carregado!")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {str(e)}")
        return
    
    # Configurar LoRA para treinamento eficiente
    print("\n🔧 Configurando LoRA...")
    
    peft_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Split train/test
    print("\n📊 Dividindo dataset...")
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    print(f"   Train: {len(train_dataset)} pares")
    print(f"   Test: {len(eval_dataset)} pares")
    
    # Configuração de treinamento DPO
    print("\n⚙️  Configurando treinamento...")
    
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",  # Pode mudar para "tensorboard" se quiser
    )
    
    # Criar trainer
    print("\n🎯 Criando DPO Trainer...")
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # Parâmetro de temperatura DPO
    )
    
    # Treinar
    print("\n🚀 Iniciando treinamento...")
    print(f"   Épocas: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print("\n" + "-"*60)
    
    try:
        trainer.train()
        
        print("\n" + "-"*60)
        print("✅ Treinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Salvar modelo
    print(f"\n💾 Salvando modelo em: {output_dir}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Modelo salvo!")
    
    # Instruções de uso
    print("\n" + "="*60)
    print("🎉 TREINAMENTO CONCLUÍDO!")
    print("="*60)
    print(f"\n📍 Como usar o modelo treinado:")
    print(f"\n1. Atualize config/settings.py:")
    print(f"   MODEL_NAME = \"{output_dir}\"")
    print(f"\n2. Execute o chatbot normalmente:")
    print(f"   python main_with_feedback.py")
    print(f"\n3. Compare com o modelo original e continue coletando feedback!")
    print("\n" + "="*60 + "\n")


def main():
    """
    Função principal
    """
    print("\n" + "="*60)
    print("🎯 DPO TRAINING - GALILEU GALILEI")
    print("="*60)
    
    print("\n⚠️  REQUISITOS:")
    print("   - Pelo menos 50 pares de preferência")
    print("   - GPU recomendada (16GB+ VRAM)")
    print("   - ~2-4 horas de treinamento")
    print("   - ~20GB de espaço em disco")
    
    # Verificar se quer continuar
    response = input("\nDeseja continuar? (s/n): ")
    if response.lower() != 's':
        print("Treinamento cancelado.")
        return
    
    try:
        # Preparar dataset
        dataset = prepare_dpo_dataset(min_samples=50)
        
        # Treinar modelo
        train_dpo_model(dataset)
        
    except ValueError as e:
        print(f"\n⚠️  {str(e)}")
        print("\nPróximos passos:")
        print("   1. Continue usando main_with_feedback.py")
        print("   2. Colete mais feedback de usuários")
        print("   3. Tente gerar múltiplas respostas para mesmas queries")
        print("   4. Execute este script novamente quando tiver dados suficientes")
        
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
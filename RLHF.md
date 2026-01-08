# 🎯 Guia Completo: Implementando RLHF no seu Projeto

## 📋 Índice

1. [O que é RLHF?](#o-que-é-rlhf)
2. [Arquitetura e Fluxo](#arquitetura-e-fluxo)
3. [Nível 1: Sistema de Feedback Simples](#nível-1-sistema-de-feedback-simples) ✅ IMPLEMENTADO
4. [Nível 2: DPO (Direct Preference Optimization)](#nível-2-dpo-direct-preference-optimization)
5. [Nível 3: RLHF Completo (PPO)](#nível-3-rlhf-completo-ppo)
6. [Comparação das Abordagens](#comparação-das-abordagens)
7. [Recursos Necessários](#recursos-necessários)

---

## 🧠 O que é RLHF?

**RLHF (Reinforcement Learning from Human Feedback)** é uma técnica para alinhar modelos de linguagem com preferências humanas através de três fases principais:

### Fase 1: Supervised Fine-Tuning (SFT)
- Treinar o modelo base com exemplos de alta qualidade
- Criar um modelo inicial bem comportado

### Fase 2: Reward Model Training
- Coletar pares de respostas (boa vs ruim) para mesmas perguntas
- Treinar um modelo que prevê qual resposta humanos preferem
- Esse modelo aprende a "pontuar" respostas

### Fase 3: RL Fine-Tuning (PPO)
- Usar o Reward Model para guiar o treinamento
- Aplicar PPO (Proximal Policy Optimization)
- Modelo aprende a gerar respostas com scores altos

---

## 🏗️ Arquitetura e Fluxo

```
┌─────────────────────────────────────────────────────────┐
│                    FASE 1: SFT                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │  Modelo  │ -> │  + Data  │ -> │  Modelo  │        │
│  │   Base   │    │  Curada  │    │    SFT   │        │
│  └──────────┘    └──────────┘    └──────────┘        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              FASE 2: REWARD MODEL                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │ Pares de │ -> │ Treinar  │ -> │  Reward  │        │
│  │Preferên. │    │   RM     │    │  Model   │        │
│  └──────────┘    └──────────┘    └──────────┘        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                FASE 3: RL TRAINING                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │ Modelo   │ -> │   PPO    │ -> │  Modelo  │        │
│  │   SFT    │    │  + RM    │    │  Final   │        │
│  └──────────┘    └──────────┘    └──────────┘        │
└─────────────────────────────────────────────────────────┘
```

---

## 🟢 Nível 1: Sistema de Feedback Simples

### ✅ Status: IMPLEMENTADO

Você já tem isso funcionando com os arquivos:
- `src/feedback/feedback_collector.py` - Coleta e armazena feedback
- `main_with_feedback.py` - Interface Gradio com thumbs up/down

### Como usar:

```bash
# Execute a interface com feedback
python main_with_feedback.py
```

### O que foi implementado:

1. **Coleta de Feedback:**
   - Thumbs up/down
   - Ratings (1-5 estrelas)
   - Comentários textuais
   - Armazenamento em SQLite

2. **Análise:**
   - Estatísticas de feedback
   - Export para treinamento futuro
   - Pares de preferência

### Próximos passos:

```python
# Exportar dados coletados
from src.feedback.feedback_collector import FeedbackCollector

collector = FeedbackCollector()

# Exportar respostas com alta avaliação
collector.export_for_training(
    "data/feedback/training_data.jsonl",
    min_rating=4
)

# Exportar pares de preferência
collector.export_preference_pairs(
    "data/feedback/preference_pairs.json"
)
```

---

## 🟡 Nível 2: DPO (Direct Preference Optimization)

### O que é DPO?

DPO é uma alternativa **mais simples** ao RLHF tradicional:
- ✅ Não precisa de Reward Model separado
- ✅ Mais estável e fácil de implementar
- ✅ Requer menos recursos computacionais
- ✅ Funciona diretamente com pares de preferência

### Implementação com TRL (Transformers Reinforcement Learning)

#### 1. Instalação:

```bash
pip install trl>=0.7.0
pip install datasets>=2.14.0
pip install peft>=0.6.0
```

#### 2. Preparar dados de preferência:

```python
# dpo_data_preparation.py
import json
from datasets import Dataset

def prepare_dpo_dataset(feedback_file: str):
    """
    Converte feedback coletado para formato DPO
    """
    with open(feedback_file, 'r') as f:
        feedbacks = json.load(f)
    
    dpo_data = []
    
    # Agrupar por query para encontrar pares
    from collections import defaultdict
    by_query = defaultdict(list)
    
    for fb in feedbacks:
        by_query[fb['query']].append(fb)
    
    # Criar pares: melhor resposta vs pior resposta
    for query, responses in by_query.items():
        if len(responses) < 2:
            continue
        
        # Ordenar por rating
        responses.sort(key=lambda x: x['rating'], reverse=True)
        
        # Pegar melhor e pior
        best = responses[0]
        worst = responses[-1]
        
        if best['rating'] > worst['rating']:
            dpo_data.append({
                'prompt': query,
                'chosen': best['response'],
                'rejected': worst['response']
            })
    
    # Converter para Dataset
    dataset = Dataset.from_list(dpo_data)
    return dataset

# Usar
dataset = prepare_dpo_dataset('data/feedback/preference_pairs.json')
dataset.save_to_disk('data/dpo/dataset')
print(f"✅ {len(dataset)} pares preparados para DPO")
```

#### 3. Script de treinamento DPO:

```python
# dpo_training.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

# ===== CONFIGURAÇÃO =====
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "models/galileu_dpo"

# Carregar modelo e tokenizer
print("📥 Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Configurar LoRA para treinamento eficiente
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Aplicar LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Carregar dataset
print("📂 Carregando dataset...")
dataset = load_from_disk('data/dpo/dataset')
train_dataset = dataset.train_test_split(test_size=0.1)

# Configuração DPO
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    fp16=True,
    remove_unused_columns=False,
)

# Criar trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset['train'],
    eval_dataset=train_dataset['test'],
    tokenizer=tokenizer,
    beta=0.1,  # Parâmetro de temperatura DPO
)

# Treinar
print("🚀 Iniciando treinamento DPO...")
trainer.train()

# Salvar modelo
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modelo DPO salvo em: {OUTPUT_DIR}")
```

#### 4. Usar modelo treinado:

```python
# Atualizar config/settings.py
MODEL_NAME = "models/galileu_dpo"  # Seu modelo fine-tuned

# Executar normalmente
python main_with_feedback.py
```

### Vantagens do DPO:

- ✅ Mais simples que RLHF completo
- ✅ Não precisa de Reward Model separado
- ✅ Treinamento mais estável
- ✅ Funciona bem com poucos dados (500-1000 pares)

---

## 🔴 Nível 3: RLHF Completo (PPO)

### Quando usar RLHF completo?

- Você tem **muitos dados** (10k+ interações)
- Precisa de **controle fino** sobre o reward
- Tem **recursos computacionais** (múltiplas GPUs)
- Quer **state-of-the-art** results

### Implementação com TRL

#### 1. Treinar Reward Model:

```python
# reward_model_training.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from trl import RewardTrainer, RewardConfig

# ===== CONFIGURAÇÃO =====
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "models/reward_model"

# Carregar modelo para classificação (reward scoring)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,  # Score contínuo
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Preparar dados
def prepare_reward_data(feedback_file):
    """
    Formato: query + response -> score
    """
    with open(feedback_file) as f:
        feedbacks = json.load(f)
    
    data = []
    for fb in feedbacks:
        if fb['rating'] is not None:
            # Normalizar rating para 0-1
            score = fb['rating'] / 5.0
            
            data.append({
                'text': f"Query: {fb['query']}\nResponse: {fb['response']}",
                'label': score
            })
    
    return Dataset.from_list(data)

dataset = prepare_reward_data('data/feedback/training_data.jsonl')
train_test = dataset.train_test_split(test_size=0.1)

# Configuração de treinamento
training_args = RewardConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    fp16=True,
)

# Trainer
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_test['train'],
    eval_dataset=train_test['test'],
    tokenizer=tokenizer,
)

# Treinar
print("🚀 Treinando Reward Model...")
trainer.train()

# Salvar
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Reward Model salvo em: {OUTPUT_DIR}")
```

#### 2. Treinar com PPO:

```python
# ppo_training.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import load_dataset

# ===== CONFIGURAÇÃO =====
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
REWARD_MODEL = "models/reward_model"
OUTPUT_DIR = "models/galileu_ppo"

# Carregar modelo policy
print("📥 Carregando modelo policy...")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Carregar reward model
print("📥 Carregando reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL,
    num_labels=1,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configuração PPO
config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    early_stopping=True,
    target_kl=0.1,
    ppo_epochs=4,
)

# Criar trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
)

# Preparar queries de treinamento
queries = [
    "Quando Galileu nasceu?",
    "Quais foram suas descobertas?",
    "O que aconteceu com a Igreja?",
    # ... adicionar mais queries
]

# Loop de treinamento PPO
print("🚀 Iniciando treinamento PPO...")

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
}

for epoch in range(3):
    for batch_idx, query in enumerate(queries):
        # Tokenizar query
        query_tensors = tokenizer.encode(query, return_tensors="pt").to(model.device)
        
        # Gerar resposta
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        
        response_text = tokenizer.decode(response_tensors[0])
        
        # Calcular reward usando reward model
        input_text = f"Query: {query}\nResponse: {response_text}"
        inputs = tokenizer(input_text, return_tensors="pt").to(reward_model.device)
        
        with torch.no_grad():
            reward_score = reward_model(**inputs).logits[0].item()
        
        reward = torch.tensor([reward_score]).to(model.device)
        
        # PPO step
        stats = ppo_trainer.step([query_tensors[0]], [response_tensors[0]], [reward])
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Reward = {reward_score:.4f}")

# Salvar modelo final
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modelo PPO salvo em: {OUTPUT_DIR}")
```

---

## 📊 Comparação das Abordagens

| Aspecto | Feedback Simples | DPO | RLHF (PPO) |
|---------|------------------|-----|------------|
| **Complexidade** | 🟢 Baixa | 🟡 Média | 🔴 Alta |
| **Dados necessários** | 100+ feedbacks | 500-1000 pares | 10k+ interações |
| **Recursos computacionais** | Mínimos | 1 GPU (16GB+) | Múltiplas GPUs |
| **Tempo de implementação** | 1 dia | 1 semana | 2-4 semanas |
| **Qualidade dos resultados** | Análise | Boa | Excelente |
| **Estabilidade** | N/A | 🟢 Alta | 🟡 Média |
| **Custo** | ~$0 | ~$50-200 | ~$500-2000 |

---

## 💻 Recursos Necessários

### Para DPO:
```
GPU: 16GB VRAM mínimo (RTX 4080, A100)
RAM: 32GB
Tempo: ~2-4 horas de treinamento
Custo (cloud): ~$10-50
```

### Para RLHF Completo:
```
GPU: 40GB+ VRAM (A100, H100)
RAM: 64GB+
Tempo: ~8-24 horas de treinamento
Custo (cloud): ~$200-1000
```

### Alternativas mais baratas:

1. **LoRA + Quantização:**
   ```python
   # Treinar com 4-bit
   from transformers import BitsAndBytesConfig
   
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16,
   )
   ```

2. **Usar modelos menores:**
   - Llama-3.1-3B em vez de 8B
   - Phi-3-mini (3.8B)
   - Gemma-2B

3. **Cloud Computing:**
   - Google Colab Pro ($10/mês) - GPU T4/A100
   - Paperspace Gradient - GPU por hora
   - Lambda Labs - GPUs dedicadas

---

## 🎯 Recomendação para seu Projeto

### Começar com (FEITO ✅):
1. Sistema de feedback simples
2. Coletar 500-1000 feedbacks de qualidade

### Próximo passo (Recomendado 🟡):
1. Implementar DPO
2. É mais simples e efetivo que RLHF completo
3. Requer menos recursos

### Futuro (Opcional 🔴):
1. Se tiver muitos dados e recursos
2. Implementar RLHF completo com PPO
3. Para máxima qualidade

---

## 📚 Recursos Adicionais

### Papers:
- [InstructGPT (RLHF Original)](https://arxiv.org/abs/2203.02155)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

### Bibliotecas:
- [TRL (Transformers RL)](https://github.com/huggingface/trl)
- [TRLX](https://github.com/CarperAI/trlx)
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

### Tutoriais:
- [Hugging Face DPO Tutorial](https://huggingface.co/docs/trl/dpo_trainer)
- [Hugging Face PPO Tutorial](https://huggingface.co/docs/trl/ppo_trainer)

---

## 🎬 Próximos Passos Práticos

1. **Execute o sistema de feedback:**
   ```bash
   python main_with_feedback.py
   ```

2. **Colete feedbacks reais:**
   - Use o chatbot normalmente
   - Peça para amigos/colegas testarem
   - Avalie cada resposta

3. **Analise os dados:**
   ```python
   from src.feedback.feedback_collector import FeedbackCollector
   collector = FeedbackCollector()
   stats = collector.get_feedback_stats()
   print(stats)
   ```

4. **Quando tiver 500+ feedbacks:**
   - Exporte os dados
   - Implemente DPO
   - Treine o modelo

5. **Integre o modelo treinado:**
   - Atualize `MODEL_NAME` em `config/settings.py`
   - Teste e compare com o modelo original

---

**Boa sorte com a implementação! 🚀**

Se precisar de ajuda, abra uma issue ou entre em contato!
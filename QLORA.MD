# 🔧 Guia de Quantização (QLoRA)

## O que é Quantização?

Quantização reduz a precisão dos pesos do modelo (de 16-bit para 4-bit), diminuindo drasticamente o uso de memória com perda mínima de qualidade.

## 📊 Comparação de Uso de Memória

| Configuração | VRAM Necessária | Qualidade | Velocidade |
|--------------|-----------------|-----------|------------|
| **FP16 (sem quantização)** | ~16GB | 100% | Rápida |
| **8-bit (INT8)** | ~8GB | ~98% | Média |
| **4-bit (NF4/QLoRA)** | ~4GB | ~95% | Lenta |

### 🎯 Para RTX 3050 (4GB VRAM)

**Recomendado:** Use **4-bit quantização** (QLoRA)
- ✅ Cabe perfeitamente em 4GB
- ✅ Qualidade ainda excelente (~95% do original)
- ⚠️ Um pouco mais lento na geração

## 🚀 Como Usar

### 1. Instalar bitsandbytes
```bash
pip install bitsandbytes>=0.41.0
```

**Nota:** bitsandbytes requer CUDA. Não funciona apenas com CPU.

### 2. Configurar .env
```bash
USE_QUANTIZATION=True
QUANTIZATION_BITS=4  # 4-bit para 4GB VRAM, 8-bit para 8GB+
```

### 3. Executar normalmente
```bash
python main.py
```

## 📋 Tipos de Quantização

### 4-bit (NF4 - Normal Float 4)

**Melhor para:** GPUs com 4-6GB VRAM (como RTX 3050)

**Características:**
- 🎯 Uso de memória: ~4GB
- ⭐ Qualidade: ~95% do modelo original
- ✅ Double Quantization ativa (nested quantization)

### 8-bit (INT8)

**Melhor para:** GPUs com 8-10GB VRAM (como RTX 3060)

**Características:**
- 🎯 Uso de memória: ~8GB
- ⭐ Qualidade: ~98% do modelo original

## 🔬 Detalhes Técnicos
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

## 🧪 Benchmarks — Llama-3.2-3B-Instruct

| Métrica | FP16 | 8-bit | 4-bit NF4 |
|---------|------|-------|-----------|
| VRAM | 16GB | 8GB | 4GB |
| Tokens/seg | 100 | 85 | 70 |
| Qualidade | 100% | 98% | 95% |

## ⚠️ Troubleshooting

**"CUDA out of memory"** → Use `QUANTIZATION_BITS=4` ou reduza `max_new_tokens`

**"bitsandbytes not found"**
```bash
pip install bitsandbytes>=0.41.0 --force-reinstall
```

**Windows: "Could not find bitsandbytes CUDA binary"**
```bash
pip uninstall bitsandbytes
pip install bitsandbytes>=0.41.0
```
Se persistir, baixe o wheel manualmente: https://github.com/jllllll/bitsandbytes-windows-webui/releases

## 📚 Referências

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main_classes/quantization)
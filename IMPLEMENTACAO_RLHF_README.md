# 🎯 Sistema de Feedback e RLHF - Implementação Completa

## 📦 O que foi implementado?

Implementei um **sistema completo de coleta de feedback** e preparei seu projeto para **RLHF (Reinforcement Learning from Human Feedback)** com três níveis de implementação:

### ✅ Nível 1: Sistema de Feedback (IMPLEMENTADO)
- Coleta de thumbs up/down
- Sistema de ratings (1-5 estrelas)
- Comentários dos usuários
- Armazenamento em SQLite
- Análise e visualização de dados
- Interface Gradio atualizada

### 🟡 Nível 2: DPO - Direct Preference Optimization (PRONTO PARA USO)
- Script de preparação de dados
- Script de treinamento DPO
- Mais simples que RLHF completo
- Requer 500-1000 pares de preferência

### 🔴 Nível 3: RLHF Completo com PPO (GUIA INCLUÍDO)
- Guia completo de implementação
- Training de Reward Model
- PPO (Proximal Policy Optimization)
- Para quando tiver 10k+ interações

---

## 📁 Arquivos Criados

```
src/feedback/
├── __init__.py
├── feedback_collector.py      # Sistema de coleta de feedback
└── feedback_analyzer.py        # Análise e visualização

main_with_feedback.py          # Interface Gradio + Feedback
train_dpo.py                   # Script de treinamento DPO
RLHF_GUIDE.md                  # Guia completo de RLHF
```

---

## 🚀 Como Usar

### 1. Execute a interface com sistema de feedback:

```bash
python main_with_feedback.py
```

**Funcionalidades da nova interface:**
- Chat normal com Galileu
- 👍 Botão de Thumbs Up
- 👎 Botão de Thumbs Down
- ⭐ Sistema de rating 1-5
- ✍️ Campo para comentários
- 📊 Estatísticas de feedback
- 📈 Estatísticas do sistema

### 2. Colete feedback dos usuários:

- Use o chatbot normalmente
- Avalie cada resposta com thumbs up/down
- Opcionalmente, dê ratings e comentários detalhados
- Peça para outras pessoas testarem e avaliarem

**Meta:** Coletar pelo menos 500 feedbacks para treinar com DPO

### 3. Analise o feedback coletado:

```bash
python -c "
from src.feedback.feedback_analyzer import FeedbackAnalyzer
analyzer = FeedbackAnalyzer()
analyzer.generate_report('data/feedback/report.json')
"
```

Ou diretamente:

```bash
python src/feedback/feedback_analyzer.py
```

**O que você verá:**
- Estatísticas gerais
- Análise de sentimento
- Qualidade das respostas
- Queries mais comuns
- Problemas identificados
- Recomendações de melhoria

### 4. Quando tiver dados suficientes, treine com DPO:

```bash
python train_dpo.py
```

**Requisitos para DPO:**
- Mínimo 50 pares de preferência (recomendado 500+)
- GPU com 16GB+ VRAM (ou use Colab)
- 2-4 horas de treinamento

### 5. Use o modelo treinado:

Após o treinamento, atualize `config/settings.py`:

```python
# De:
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Para:
MODEL_NAME = "models/galileu_dpo"
```

Depois execute normalmente:

```bash
python main_with_feedback.py
```

---

## 📊 Estrutura de Dados

### Feedback armazenado em SQLite:

**Tabela `feedback`:**
- `timestamp`: Data/hora do feedback
- `session_id`: ID da sessão
- `query`: Pergunta do usuário
- `response`: Resposta do chatbot
- `rating`: Rating 1-5 (opcional)
- `thumbs_up`: True/False (opcional)
- `comment`: Comentário do usuário (opcional)
- `source_documents`: Documentos usados (JSON)
- `metadata`: Metadados adicionais (JSON)

**Tabela `preference_pairs`:**
- `query`: Pergunta
- `response_chosen`: Resposta melhor
- `response_rejected`: Resposta pior
- `reason`: Razão da escolha
- `metadata`: Metadados

**Localização do banco:** `data/feedback/feedback.db`

---

## 🎯 Fluxo Recomendado

### Fase 1: Coleta de Feedback (Semanas 1-4)
```
1. Execute main_with_feedback.py
2. Use o chatbot normalmente
3. Avalie todas as respostas
4. Compartilhe com amigos/colegas para testar
5. Meta: 500+ feedbacks
```

### Fase 2: Análise (Semana 5)
```
1. Execute feedback_analyzer.py
2. Revise estatísticas e problemas
3. Identifique padrões
4. Ajuste prompts se necessário
5. Exporte dados para treinamento
```

### Fase 3: Treinamento DPO (Semana 6)
```
1. Verifique se tem 500+ pares
2. Execute train_dpo.py
3. Aguarde treinamento (2-4h)
4. Teste modelo treinado
5. Compare com modelo original
```

### Fase 4: Iteração (Contínuo)
```
1. Continue coletando feedback com novo modelo
2. Analise melhorias
3. Re-treine periodicamente
4. Ciclo de melhoria contínua
```

---

## 📈 Monitoramento

### Verificar estatísticas na interface:

1. Abra `main_with_feedback.py`
2. Vá para a aba "📊 Estatísticas"
3. Clique em "Atualizar Estatísticas de Feedback"

### Via código:

```python
from src.feedback.feedback_collector import FeedbackCollector

collector = FeedbackCollector()
stats = collector.get_feedback_stats()
print(stats)
```

### Exportar dados:

```python
# Exportar feedbacks positivos para análise
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

## 🆘 Troubleshooting

### "Poucos dados para DPO"
**Solução:** Continue coletando feedback. Você precisa de pelo menos 50 pares de preferência (respostas diferentes para mesmas queries com ratings diferentes).

### "GPU out of memory durante treinamento"
**Soluções:**
1. Reduza `batch_size` em `train_dpo.py`
2. Use quantização 4-bit
3. Use Google Colab com GPU gratuita
4. Considere cloud computing (Lambda Labs, etc)

### "Erro ao importar TRL"
**Solução:**
```bash
pip install trl>=0.7.0
pip install peft>=0.6.0
pip install datasets>=2.14.0
```

### "Modelo treinado não funciona"
**Verificações:**
1. Conferir se `MODEL_NAME` está correto em `settings.py`
2. Verificar se modelo foi salvo corretamente
3. Testar com modelo original primeiro
4. Revisar logs de treinamento

---

## 💡 Dicas e Boas Práticas

### Para Coletar Feedback de Qualidade:

1. **Seja consistente:** Avalie todas as respostas, não só as ruins
2. **Use comentários:** Explique por que deu aquela nota
3. **Teste edge cases:** Perguntas difíceis ou ambíguas
4. **Diversifique:** Faça perguntas sobre diferentes aspectos de Galileu
5. **Seja honesto:** Avalie objetivamente, não seja gentil demais

### Para Melhor Treinamento DPO:

1. **Diversidade de queries:** Diferentes tipos de perguntas
2. **Pares claros:** Diferença óbvia entre boa e má resposta
3. **Volume:** Quanto mais dados, melhor (500-1000 ideal)
4. **Qualidade > Quantidade:** Prefira menos dados de qualidade
5. **Balance:** Mix de queries fáceis e difíceis

### Para Iteração Contínua:

1. **Monitore métricas:** Acompanhe rating médio ao longo do tempo
2. **A/B Testing:** Compare modelo novo vs antigo
3. **Feedback loop:** Use modelo melhorado para gerar mais dados
4. **Documente mudanças:** Anote o que funcionou/não funcionou
5. **Compartilhe resultados:** Mostre melhorias para motivar

---

## 📚 Recursos Adicionais

### Documentação:
- [RLHF_GUIDE.md](./RLHF_GUIDE.md) - Guia completo de RLHF
- [TRL Documentation](https://huggingface.co/docs/trl/) - Transformers RL
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Artigo original

### Ferramentas:
- [LangChain](https://python.langchain.com/) - Framework RAG
- [Hugging Face](https://huggingface.co/) - Modelos e datasets
- [Gradio](https://gradio.app/) - Interface web

### Comunidades:
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [LangChain Discord](https://discord.gg/langchain)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)

---

## 🎬 Próximos Passos Imediatos

1. ✅ **Instale dependências do feedback** (já incluídas)
   ```bash
   # Já está em requirements.txt
   ```

2. ✅ **Execute a nova interface**
   ```bash
   python main_with_feedback.py
   ```

3. ✅ **Teste o sistema de feedback**
   - Faça algumas perguntas
   - Avalie as respostas
   - Verifique se dados estão sendo salvos

4. ✅ **Compartilhe com outros**
   - Peça feedback de amigos/colegas
   - Colete pelo menos 50-100 avaliações iniciais

5. ✅ **Analise os primeiros dados**
   ```bash
   python src/feedback/feedback_analyzer.py
   ```

6. ✅ **Continue coletando até ter 500+**
   - Meta: 500-1000 feedbacks
   - Depois: Treinar com DPO

---

## 🤝 Suporte

Se tiver dúvidas ou problemas:

1. Revise `RLHF_GUIDE.md` para detalhes técnicos
2. Verifique os exemplos de código nos scripts
3. Consulte a documentação das bibliotecas
4. Abra uma issue no GitHub (se aplicável)

---

## 📝 Licença

Este projeto é de uso educacional. Siga as licenças dos modelos e bibliotecas utilizados.

---

**Desenvolvido para melhorar continuamente o Chatbot Galileu Galilei! 🔭✨**

Boa sorte com a implementação! 🚀

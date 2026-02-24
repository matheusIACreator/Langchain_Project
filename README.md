---
title: Science Chat API
emoji: 🔭
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
short_description: Multi-figure RAG API — chat with Galileo, Newton & Einstein
---

# 🔭 Science Chat — Historical Figures RAG API

Backend API powering **Science Chat**, a RAG system that lets you have conversations with historical scientists.

## What it does

Answers questions about Galileo Galilei, Isaac Newton, and Albert Einstein using hybrid retrieval (BM25 + semantic search) and Llama 3.2 3B Instruct. Supports any language — ask in English, get an answer in English. Ask in Italian, get an answer in Italian.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/figures` | List available figures |
| POST | `/chat` | Send a message |
| DELETE | `/conversation/{session_id}` | Clear session history |

## Example
```bash
curl -X POST https://shinigami4242557-science-chat-api.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "When was Newton born?"}'
```

## Frontend

Live at → [langchain-project-nine.vercel.app](https://langchain-project-nine.vercel.app)

## Stack

FastAPI · LangChain · ChromaDB · BM25 · Llama 3.2 3B · HuggingFace

---

Built by [Matheus Masago](https://github.com/matheusIACreator)

---
license: mit
---
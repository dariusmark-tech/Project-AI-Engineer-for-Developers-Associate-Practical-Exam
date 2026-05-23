# 🤖 Automating Customer Support with OpenAI API

> **AI Engineer for Developers Associate — Practical Exam**
> Built with Python · OpenAI Embeddings · GPT-3.5-turbo · Cosine Similarity

---

## 📌 Overview

This project implements an **automated customer support chatbot** for **ChatSolveAI**, a company that uses AI to improve response times and accuracy for customer queries. The solution leverages OpenAI's GPT models to classify queries, retrieve relevant responses, and log interactions in a structured way.

The project is divided into three tasks:

| Task | Description |
|------|-------------|
| **Task 1** | Generate and store text embeddings from a knowledge base |
| **Task 2** | Perform similarity search to match customer queries with predefined responses |
| **Task 3** | Build a full chatbot with conversation history and GPT fallback |

---

## 🗂️ Project Structure

```
├── knowledge_base.csv              # Source knowledge base (products, services, policies)
├── processed_queries.csv           # Preprocessed customer queries
├── predefined_responses.json       # Predefined responses for similarity matching
├── chatbot_responses.json          # Predefined chatbot responses (Task 3)
├── knowledge_embeddings.json       # Output: Task 1 embeddings
├── query_responses.json            # Output: Task 2 similarity results
├── sample_chatbot_responses.json   # Output: Task 3 chatbot interaction log
└── notebook.ipynb                  # Main solution notebook
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.8+
- An OpenAI API key

### Installation

```bash
pip install openai numpy pandas
```

### Environment

```python
import os
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY environment variable
model = "gpt-3.5-turbo"
```

---

## 🧩 Task 1 — Knowledge Base Embeddings

**Goal:** Convert the knowledge base into embedding vectors for efficient semantic retrieval.

**Approach:**
- Load `knowledge_base.csv`
- Embed all `document_text` values in a **single batch API request** using `text-embedding-3-small`
- No text transformations applied (no lowercasing, stripping, or normalization)
- Store results in `knowledge_embeddings.json`

**Output format:**
```json
[
  {
    "document_id": 1,
    "document_text": "Example document text.",
    "embedding_vector": [0.123, 0.456, ...],
    "metadata": "Additional document info"
  }
]
```

**Key design decision:** All texts are embedded in one API call (passing a list), avoiding per-item loops and reducing rate limit risk.

---

## 🔍 Task 2 — Similarity Search & Query Matching

**Goal:** Match customer queries to the most relevant predefined responses using cosine similarity.

**Approach:**
- Load `processed_queries.csv` and `predefined_responses.json`
- Embed both query texts and predefined response texts in **single batch requests**
- Compute cosine similarity between each query and all predefined responses
- Return the **top 3 most relevant responses** per query
- Scale confidence scores to **[0, 1]** using min-max normalization
- Implement **retry with exponential back-off** for rate limit handling
- Store results in `query_responses.json`

**Output format:**
```json
[
  {
    "query_id": 1,
    "query_text": "How do I cancel my subscription?",
    "top_responses": ["Response A", "Response B", "Response C"],
    "confidence_scores": [0.98, 0.76, 0.54]
  }
]
```

**Cosine similarity formula:**

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

---

## 💬 Task 3 — Conversational Chatbot

**Goal:** Build a full chatbot that retrieves predefined responses or generates new ones when no match is found.

**Approach:**
- Embed all predefined responses from `chatbot_responses.json` on startup
- For each query, compute semantic similarity and find the best match
- If the best cosine score is **below a threshold (0.75)** → fall back to **GPT-3.5-turbo** generation
- Log every interaction with timestamp, response, and confidence score
- Store conversation history in `sample_chatbot_responses.json`

**Test queries included:**
- 🔁 **Paraphrased query** — *"When can I talk to someone from support?"* — tests semantic similarity matching
- ❓ **Open-ended query** — *"What is your refund policy?"* — triggers GPT fallback generation

**Output format:**
```json
[
  {
    "query_text": "When can I talk to someone from support?",
    "retrieved_response": "Our support team is available from 9 AM to 5 PM, Monday to Friday.",
    "timestamp": "2025-04-02T14:30:00Z",
    "confidence_score": 0.92
  }
]
```

---

## 🛡️ Error Handling

All API calls implement **retry with exponential back-off**:

```python
def embed_texts(texts, retries=5):
    for attempt in range(retries):
        try:
            response = client.embeddings.create(input=texts, model="text-embedding-3-small")
            return [item.embedding for item in response.data]
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(2 ** attempt)  # 1s → 2s → 4s → 8s → 16s
            else:
                raise
```

---

## 📊 Key Concepts Used

- **Text Embeddings** — Dense vector representations of text using `text-embedding-3-small`
- **Cosine Similarity** — Measures semantic closeness between query and response vectors
- **Min-Max Normalization** — Scales raw similarity scores to a consistent [0, 1] range
- **Batch API Requests** — Multiple texts embedded in a single call for efficiency
- **Exponential Back-off** — Retry strategy to gracefully handle API rate limits
- **RAG Pattern** — Retrieve-then-generate: use retrieval first, fall back to generation

---

## 📝 Notes

- The OpenAI Embeddings API accepts a **list of strings** in a single request — always batch when possible
- Do **not** apply text transformations (lowercase, strip) before embedding — preserve original text
- Confidence scores are **scaled per-query** using min-max normalization across all candidate similarities
- GPT fallback is triggered when cosine similarity falls below **0.75**

---

## 📄 License

This project was completed as part of the **DataCamp AI Engineer for Developers Associate Practical Exam**.

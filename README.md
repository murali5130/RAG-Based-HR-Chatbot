# RAG-Based HR Chatbot

A chatbot that answers employee questions about HR policies using **Retrieval-Augmented Generation (RAG)**. Instead of relying on the model's general knowledge, it retrieves the relevant sections of your HR documents and generates answers grounded in them.

> Fields marked `<...>` are placeholders. Replace them with the details from your code before publishing.

## Features

- Answers HR policy questions (leave, attendance, benefits, code of conduct, etc.) from your own documents
- Retrieves the most relevant document chunks for each question before generating an answer
- `<Shows source / page reference for each answer, if implemented>`
- `<Maintains chat history, if implemented>`
- `<Web UI: Streamlit / Gradio / Flask / other>`

## How It Works

1. **Load** the HR policy documents (`<PDF / DOCX / TXT>`).
2. **Split** them into smaller text chunks.
3. **Embed** each chunk using `<embedding model>`.
4. **Store** the embeddings in `<vector database: FAISS / ChromaDB / other>`.
5. **Retrieve** the top-k chunks most similar to the user's question.
6. **Generate** the answer with `<LLM: OpenAI / Gemini / local model>`, using the retrieved chunks as context.

```
User question -> Embedding -> Vector search -> Top-k chunks -> LLM -> Answer
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | `<Python version>` |
| Framework | `<LangChain / LlamaIndex / custom>` |
| LLM | `<model name>` |
| Embeddings | `<embedding model>` |
| Vector store | `<FAISS / ChromaDB / other>` |
| UI | `<Streamlit / Gradio / Flask / other>` |

## Project Structure

```
RAG-Based-HR-Chatbot/
<paste the output of your folder tree here>
```

## Getting Started

### Prerequisites

- `<Python 3.x>`
- An API key for `<LLM provider>` (if you use a hosted model)

### Installation

```bash
git clone https://github.com/murali5130/RAG-Based-HR-Chatbot.git
cd RAG-Based-HR-Chatbot

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
<API_KEY_NAME>=your_api_key_here
```

Do not commit your `.env` file. Make sure it is listed in `.gitignore`.

### Add Your Documents

Place your HR policy files in `<folder name>/`.

### Run

```bash
<command to run, e.g. streamlit run app.py>
```

## Example Questions

- "How many casual leaves do employees get per year?"
- "What is the notice period policy?"
- "What is the process to apply for work from home?"

## Limitations

- Answers are only as good as the documents provided. If a topic is not in the documents, the chatbot may not answer correctly.
- Always verify important decisions (payroll, legal, termination) with the HR team.

## Future Improvements

- `<Add your planned improvements>`

## Author

**Murali Krishna**
GitHub: [murali5130](https://github.com/murali5130)

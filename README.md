# DebateBot

DebateBot is an earlier personal Streamlit project for asking debate questions against a local knowledge base. It retrieves a small set of relevant source lines and supplies them as context to a Groq chat completion.

## Why

During debate preparation, reference material can be scattered across notes. This project experiments with using a local text file as context for a conversational debate assistant.

## What it does

- Loads `src/debate_data.txt` when the app starts.
- Accepts a question through Streamlit's chat interface.
- Preserves the current session's chat history.
- Sends retrieved context and the user's question to Groq's `llama-3.1-8b-instant` model.

## Retrieval approach

The retrieval is intentionally simple:

1. The app fits `TfidfVectorizer` to the user's question and selects up to five high-weighted non-stopword terms.
2. It scans the knowledge base line by line and keeps the first five lines containing any selected term.
3. Those lines are placed in a `<CONTEXT>` block before the question sent to Groq.

This is keyword-based matching, not an embedding index or a vector-database RAG system.

## Tech stack

- Python
- Streamlit
- scikit-learn (`TfidfVectorizer`)
- Groq API

## Running locally

```bash
git clone https://github.com/SanjithTammana/DebateBot.git
cd DebateBot
python -m venv .venv
```

Activate the virtual environment, install dependencies, then start Streamlit:

```bash
pip install -r requirements.txt
streamlit run src/main.py
```

## Environment variables

Set the following before starting the app:

- `GROQ_API_KEY`

## Limitations

- Retrieval uses substring matches over individual lines, so it can miss relevant material that uses different wording.
- The selected lines are not ranked against the full knowledge base.
- Responses come from an LLM and should be checked against primary debate evidence before use.
- The code is concentrated in `src/main.py`; this is an earlier experiment rather than a production service.

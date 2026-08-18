# PFR Agent v1

This is the first working PFR AI agent built around the five JSON datasets you attached:

- FundingPips
- FundedNext
- FTMO
- FundedSquad
- Alpha Capital Group

It uses a local retrieval engine to search your JSON data first. If an OpenAI API key is configured, the retrieved source excerpts are sent to the model to produce a clean answer with sources.

## 1. Open a terminal in this folder

Windows PowerShell example:

```powershell
cd C:\path\to\pfr_agent_v1
```

## 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands without activating by using `.venv\Scripts\python.exe`.

## 3. Install packages

```powershell
pip install -r requirements.txt
```

## 4. Add your API key

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Open `.env` and replace:

```text
OPENAI_API_KEY=put_your_api_key_here
```

with your real API key.

Do NOT put the API key in your website frontend. It belongs on the backend only.

## 5. Run the agent in the terminal

```powershell
python agent.py
```

Try:

- Does FTMO allow news trading?
- Can I hold trades over the weekend at FundingPips?
- What is FundedSquad's max risk rule?
- What is the performance fee rule at Alpha Capital Group?
- Does FundedNext allow weekend holding?

Type `exit` to quit.

## 6. Run it as a website backend

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

Your existing frontend can POST to:

```text
http://127.0.0.1:8000/chat
```

with JSON:

```json
{
  "message": "Does FTMO allow news trading?"
}
```

The response contains:

```json
{
  "answer": "...",
  "sources": [
    {
      "firm": "FTMO",
      "title": "...",
      "url": "..."
    }
  ]
}
```

## Important behavior

The agent is deliberately conservative. It searches the supplied PFR data before answering. If the data does not support an answer, it is instructed to say that it cannot answer confidently instead of inventing a rule.

The JSON files are the source material. The agent does not automatically browse the internet in this version.

## Next PFR upgrades

After this version works:

1. Improve retrieval/chunking and add automated evaluation questions.
2. Add conversation memory.
3. Add firm/account-model filters.
4. Add comparison answers across firms.
5. Add a proper database/vector search layer.
6. Connect your scraper so changed pages update the knowledge base automatically.
7. Add citations and "last checked" metadata to the website UI.

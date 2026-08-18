from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from retriever import PFRRetriever

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT = """You are PFR Agent, an assistant for a prop-firm rules website.

Your job is to answer questions using ONLY the supplied PFR source excerpts.

These excerpts come from the user's scraped prop-firm JSON files.

IMPORTANT RULES:

1. Treat the supplied PFR excerpts as the source of truth.
2. Do NOT invent, assume, or fill gaps using outside knowledge.
3. If the supplied sources do not contain enough information, say:
   "I couldn't find enough information in the PFR database to answer that confidently."
4. If the user names a prop firm, prioritize information from that firm.
5. If multiple sources disagree, explain the disagreement instead of silently choosing one.
6. Carefully distinguish different account types, challenge phases, funded accounts,
   instant accounts, and other programs when the sources distinguish them.
7. Answer the actual question directly.
8. Keep answers clear and reasonably concise.
9. At the end, include a Sources section containing the relevant source title and URL.
10. Never claim that a rule is current unless the supplied source supports it.
"""


def remove_thinking(text: str) -> str:
    """Remove hidden/reasoning tags if the model returns them."""
    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()


def build_context(results: list[dict]) -> str:
    """Turn retrieved PFR articles into context for the AI."""
    blocks = []

    for i, result in enumerate(results, 1):
        blocks.append(
            f"""SOURCE {i}
Firm: {result['firm']}
Title: {result['title']}
URL: {result['url']}
Category: {result.get('category') or ''}
Section: {result.get('section') or ''}

Content:
{result['content']}
"""
        )

    return "\n---\n".join(blocks)


def make_client():
    """Create a MiniMax client using the OpenAI-compatible API."""
    api_key = os.getenv("MINIMAX_API_KEY")

    if not api_key:
        return None

    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install openai"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.minimax.io/v1",
    )


def answer(
    question: str,
    retriever: PFRRetriever,
    client=None,
    model: str = "MiniMax-M2.7",
):
    """Search PFR data and optionally ask MiniMax to formulate the answer."""

    results = retriever.search(question, top_k=8)

    if not results:
        return {
            "answer": (
                "I couldn't find enough information in the PFR database "
                "to answer that confidently."
            ),
            "sources": [],
            "matches": [],
        }

    context = build_context(results)

    # If MiniMax is not configured, still show the retrieved PFR information.
    if client is None:
        source_lines = "\n".join(
            f"- {r['firm']} — {r['title']} — {r['url']}"
            for r in results[:5]
        )

        text = (
            "MiniMax is not configured yet, but I found these relevant "
            "PFR sources:\n\n"
            + "\n\n".join(
                f"**{r['firm']} — {r['title']}**\n{r['content'][:1200]}"
                for r in results[:3]
            )
            + "\n\nSources:\n"
            + source_lines
        )

        return {
            "answer": text,
            "sources": results[:5],
            "matches": results,
        }

    user_prompt = f"""User question:

{question}

PFR source excerpts:

{context}

Answer the user's question using ONLY the PFR source excerpts above.

If the sources don't contain enough information, say so clearly instead
of guessing.

Include a short Sources section with the relevant source title and URL.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    answer_text = response.choices[0].message.content or ""

    answer_text = remove_thinking(answer_text)

    if not answer_text:
        answer_text = (
            "MiniMax returned an empty answer. "
            "Please try the question again."
        )

    return {
        "answer": answer_text,
        "sources": results[:5],
        "matches": results,
    }


def main():
    data_dir = Path(__file__).parent / "data"

    print("=" * 60)
    print("PFR AGENT v1")
    print("=" * 60)

    try:
        retriever = PFRRetriever(data_dir)
    except Exception as exc:
        print("\nERROR loading PFR data:")
        print(exc)
        sys.exit(1)

    client = make_client()

    model = os.getenv(
        "MINIMAX_MODEL",
        "MiniMax-M2.7",
    )

    print(f"Loaded {len(retriever.documents)} searchable chunks")
    print("Firms:", ", ".join(retriever.firms()))

    if client:
        print(f"AI mode: MiniMax ({model})")
    else:
        print("AI mode: OFF")
        print("MINIMAX_API_KEY was not found.")

    print("\nType 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            result = answer(
                question=question,
                retriever=retriever,
                client=client,
                model=model,
            )

            print("\nPFR Agent:\n")
            print(result["answer"])
            print()

        except Exception as exc:
            print("\nERROR while contacting MiniMax:")
            print(exc)
            print()


if __name__ == "__main__":
    main()

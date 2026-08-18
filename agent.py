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


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.getenv(
    "MINIMAX_MODEL",
    "MiniMax-M2.7",
)

MAX_HISTORY_MESSAGES = 16
INITIAL_RETRIEVAL = 12
FINAL_EVIDENCE = 6
MAX_VERIFICATION_ATTEMPTS = 2


FIRMS = [
    "Alpha Capital Group",
    "FTMO",
    "FundedNext",
    "FundedSquad",
    "FundingPips",
]


# ============================================================
# MAIN SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are PFR Agent, a highly reliable prop-firm research assistant.

You have two modes.

MODE 1: NORMAL CONVERSATION
If the user is simply chatting, greeting you, asking how you are, thanking
you, joking, or asking a non-PFR question, respond naturally.

Do NOT search the PFR database for ordinary conversation.

MODE 2: PFR RESEARCH
If the user asks about a prop firm's rules, accounts, trading conditions,
payouts, drawdown, risk, positions, leverage, news trading, weekend holding,
allocation, restrictions, or other factual prop-firm information, use the
supplied PFR evidence.

The PFR evidence is authoritative for PFR rule questions.

============================================================
SOURCE RULE
============================================================

Never invent a prop-firm rule.

Never use general industry knowledge to fill a missing rule.

Never use your training knowledge to replace missing PFR evidence.

If the evidence does not establish the answer, say:

"I couldn't verify that from the PFR database."

============================================================
FIRM ISOLATION
============================================================

If the user asks about one firm, use evidence from that firm only.

Never mix:

FTMO
FundedNext
FundedSquad
FundingPips
Alpha Capital Group

unless the user explicitly asks for a comparison.

============================================================
ACCOUNT / MODEL / STAGE ISOLATION
============================================================

Never combine rules from different:

- account types
- challenge models
- evaluation phases
- funded stages
- instant accounts
- retry accounts
- funded retry accounts
- account sizes
- programs

If the source applies to only one model or stage, explicitly state that.

============================================================
FOLLOW-UP QUESTIONS
============================================================

Use conversation history.

For example:

User:
"Does FundedSquad allow layering?"

User:
"What about 2 positions?"

The second question is still about FundedSquad.

Do not reset the firm simply because the user did not repeat its name.

============================================================
EXPLICIT FACT VS INFERENCE
============================================================

A simple logical conclusion is allowed when it follows directly from the
source.

Example:

Source:
"3 or more open positions on one asset are prohibited."

Valid conclusion:
"Therefore, the maximum is 2 open positions."

But do not create additional rules that are not supported.

============================================================
NUMBERS
============================================================

Treat all numbers as high-risk information.

Before stating a number verify:

- percentage
- dollar amount
- account size
- drawdown
- daily loss
- maximum loss
- position count
- lot size
- payout split
- profit target
- leverage
- time period
- date

============================================================
ANSWER STYLE
============================================================

Answer the user's actual question first.

Be direct.

Then give the necessary explanation.

Do not dump unrelated retrieved information.

============================================================
VERIFICATION
============================================================

The supplied evidence must support the actual claim.

If the evidence does not support the answer, do not pretend that it does.

Never say "I'm sure" merely because you previously gave the same answer.

============================================================
FINAL PRIORITY
============================================================

Accuracy > source fidelity > correct firm > correct account/model/stage >
correct interpretation > clarity > brevity.
"""


# ============================================================
# VERIFIER PROMPT
# ============================================================

VERIFIER_PROMPT = """
You are the independent PFR evidence verifier.

Your job is NOT to write a nice answer.

Your job is to determine whether a proposed answer is actually supported by
the supplied PFR evidence.

Check every important factual claim.

Especially check:

1. Correct prop firm.
2. Correct account/model/stage.
3. Correct numbers.
4. Correct interpretation.
5. Correct source.
6. No information imported from another firm.
7. No unsupported inference.
8. No contradiction with the supplied evidence.

Return exactly one of:

PASS

or

FAIL: <brief reason>

If the answer contains a claim that is not supported, return FAIL.

If the answer says that something could not be verified and the evidence
actually contains the answer, return FAIL.

Do not be polite.
Do not defend the previous answer.
Do not assume the previous answer is correct.
"""


# ============================================================
# HELPERS
# ============================================================

def remove_thinking(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()


def detect_firm(text: str) -> str | None:
    q = text.lower()

    aliases = [
        ("alpha capital group", "Alpha Capital Group"),
        ("alphacapitalgroup", "Alpha Capital Group"),
        ("alpha capital", "Alpha Capital Group"),
        ("funding pips", "FundingPips"),
        ("fundingpips", "FundingPips"),
        ("funded next", "FundedNext"),
        ("fundednext", "FundedNext"),
        ("funded squad", "FundedSquad"),
        ("fundedsquad", "FundedSquad"),
        ("ftmo", "FTMO"),
    ]

    for alias, firm in aliases:
        if alias in q:
            return firm

    return None


def is_comparison(question: str) -> bool:
    q = question.lower()

    phrases = [
        "compare",
        "comparison",
        "versus",
        " vs ",
        "difference between",
        "differences between",
        "which is better",
        "which has more",
        "which has the highest",
    ]

    return any(
        phrase in q
        for phrase in phrases
    )


def looks_like_pfr_question(question: str) -> bool:
    q = question.lower().strip()

    # Explicit firm mention almost certainly means PFR research.
    if detect_firm(q):
        return True

    pfr_terms = [
        "prop firm",
        "propfirm",
        "funded account",
        "funding",
        "challenge",
        "evaluation",
        "drawdown",
        "daily loss",
        "max loss",
        "maximum loss",
        "floating pnl",
        "floating p&l",
        "floating loss",
        "payout",
        "withdrawal",
        "profit split",
        "profit target",
        "allocation",
        "leverage",
        "lot size",
        "position",
        "positions",
        "trade",
        "trading",
        "news trading",
        "weekend",
        "overnight",
        "hedging",
        "layering",
        "risk rule",
        "risk limit",
        "account rule",
        "account rules",
        "allowed",
        "prohibited",
        "restriction",
        "restrictions",
    ]

    return any(
        term in q
        for term in pfr_terms
    )


def is_casual(question: str) -> bool:
    q = question.lower().strip()

    casual_phrases = [
        "how are you",
        "how r u",
        "how are u",
        "what's up",
        "whats up",
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
        "good night",
        "who are you",
        "what can you do",
    ]

    return any(
        phrase in q
        for phrase in casual_phrases
    )


def build_source_context(results: list[dict]) -> str:
    if not results:
        return "NO PFR EVIDENCE WAS FOUND."

    blocks = []

    for i, result in enumerate(results, 1):
        blocks.append(
            f"""
SOURCE {i}

Firm: {result.get('firm', '')}
Title: {result.get('title', '')}
URL: {result.get('url', '')}
Category: {result.get('category') or ''}
Section: {result.get('section') or ''}
Retrieval score: {result.get('score', '')}

CONTENT:
{result.get('content', '')}
"""
        )

    return "\n==============================\n".join(
        blocks
    )


def deduplicate_results(
    results: list[dict],
) -> list[dict]:

    output = []
    seen = set()

    for result in results:

        content = re.sub(
            r"\s+",
            " ",
            str(result.get("content", "")).lower(),
        ).strip()

        title = str(
            result.get("title", "")
        ).strip().lower()

        # Treat identical URL/content as one source.
        key = (
            result.get("url", ""),
            title,
            content[:1000],
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(result)

    return output


def get_client():
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed."
        )

    api_key = os.getenv(
        "MINIMAX_API_KEY"
    )

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url="https://api.minimax.io/v1",
    )


# ============================================================
# MODEL CALL
# ============================================================

def call_model(
    client,
    model: str,
    messages: list[dict],
) -> str:

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    if not response.choices:
        return ""

    text = response.choices[0].message.content

    return remove_thinking(
        text or ""
    )


# ============================================================
# NORMAL CONVERSATION
# ============================================================

def casual_answer(
    client,
    model: str,
    question: str,
    history: list[dict],
) -> str:

    if client is None:
        return (
            "I'm running, bro. MiniMax isn't configured right now."
        )

    messages = [
        {
            "role": "system",
            "content": """
You are PFR Agent.

This is normal conversation, not a PFR database question.

Respond naturally and briefly.

Do not mention the PFR database.
Do not say you couldn't verify the question.
Do not invent a database answer.
""",
        }
    ]

    messages.extend(
        history[-6:]
    )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    return call_model(
        client,
        model,
        messages,
    )


# ============================================================
# PFR ANSWER
# ============================================================

def generate_pfr_answer(
    client,
    model: str,
    question: str,
    history: list[dict],
    results: list[dict],
    current_firm: str | None,
    comparison: bool,
) -> str:

    context = build_source_context(
        results
    )

    if current_firm and not comparison:
        firm_instruction = f"""
CURRENT FIRM:

{current_firm}

Use evidence from {current_firm} only.
"""
    elif comparison:
        firm_instruction = """
THIS IS A COMPARISON.

Keep every firm's rules separate.
Do not merge them.
"""
    else:
        firm_instruction = """
No firm was explicitly identified.

Use the evidence carefully.
If the correct firm cannot be established, do not guess.
"""

    prompt = f"""
CURRENT USER QUESTION:

{question}

{firm_instruction}

RECENT CONVERSATION:

{history[-8:]}

PFR EVIDENCE:

{context}

============================================================

Answer the CURRENT USER QUESTION.

Rules:

- Answer directly first.
- Use only the supplied evidence for PFR facts.
- Do not use another firm's rules.
- Do not invent missing rules.
- Respect account/model/stage differences.
- Verify numbers against the evidence.
- If a simple conclusion follows directly from the source, you may state it.
- If evidence is insufficient, say:
  "I couldn't verify that from the PFR database."

For factual PFR answers, include the most relevant source title and URL.

Do not list unrelated sources.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(
        history[-8:]
    )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    return call_model(
        client,
        model,
        messages,
    )


# ============================================================
# VERIFY ANSWER
# ============================================================

def verify_answer(
    client,
    model: str,
    question: str,
    answer: str,
    results: list[dict],
    current_firm: str | None,
) -> tuple[bool, str]:

    context = build_source_context(
        results
    )

    prompt = f"""
USER QUESTION:

{question}

CURRENT FIRM:

{current_firm or "Not explicitly identified"}

PROPOSED ANSWER:

{answer}

SUPPLIED EVIDENCE:

{context}

Verify the proposed answer against the evidence.

Remember:

- A correct logical conclusion is allowed.
- Unsupported claims are not allowed.
- Wrong firm = FAIL.
- Wrong account/model/stage = FAIL.
- Wrong number = FAIL.
- Missing evidence = FAIL.
- If the proposed answer says "couldn't verify" but the evidence clearly
  contains the answer = FAIL.
- Do not use outside knowledge.

Return exactly:

PASS

or:

FAIL: reason
"""

    result = call_model(
        client,
        model,
        [
            {
                "role": "system",
                "content": VERIFIER_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    cleaned = result.strip()

    if cleaned.upper().startswith("PASS"):
        return True, cleaned

    return False, cleaned


# ============================================================
# PFR ENGINE
# ============================================================

def answer_pfr(
    question: str,
    retriever: PFRRetriever,
    client,
    model: str,
    history: list[dict],
    current_firm: str | None,
) -> dict:

    explicit_firm = detect_firm(
        question
    )

    if explicit_firm:
        current_firm = explicit_firm

    comparison = is_comparison(
        question
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Every PFR question gets a fresh retrieval.
    # We never trust an old answer as evidence.
    # --------------------------------------------------------

    retrieval_query = question

    if current_firm and not comparison:
        retrieval_query = (
            f"{current_firm} {question}"
        )

    results = retriever.search(
        retrieval_query,
        top_k=INITIAL_RETRIEVAL,
        firm=current_firm
        if current_firm and not comparison
        else None,
    )

    results = deduplicate_results(
        results
    )

    results = results[
        :FINAL_EVIDENCE
    ]

    # --------------------------------------------------------
    # No evidence.
    # --------------------------------------------------------

    if not results:

        return {
            "answer": (
                "I couldn't verify that from the PFR database."
            ),
            "current_firm": current_firm,
            "sources": [],
        }

    # --------------------------------------------------------
    # Generate + verify.
    # --------------------------------------------------------

    if client is None:

        return {
            "answer": (
                "MiniMax is not configured, "
                "but I found these PFR sources:\n\n"
                + "\n".join(
                    f"- {r['title']} — {r['url']}"
                    for r in results
                )
            ),
            "current_firm": current_firm,
            "sources": results,
        }

    last_answer = ""

    for attempt in range(
        MAX_VERIFICATION_ATTEMPTS
    ):

        last_answer = generate_pfr_answer(
            client=client,
            model=model,
            question=question,
            history=history,
            results=results,
            current_firm=current_firm,
            comparison=comparison,
        )

        if not last_answer:
            continue

        verified, verification_message = verify_answer(
            client=client,
            model=model,
            question=question,
            answer=last_answer,
            results=results,
            current_firm=current_firm,
        )

        if verified:
            return {
                "answer": last_answer,
                "current_firm": current_firm,
                "sources": results,
            }

        # ----------------------------------------------------
        # Verification failed.
        #
        # Force another retrieval pass.
        # ----------------------------------------------------

        if attempt == 0:

            retry_query = (
                f"{question} "
                "exact rule maximum limit allowed prohibited "
                "account model stage"
            )

            if current_firm:
                retry_query = (
                    f"{current_firm} "
                    f"{retry_query}"
                )

            retry_results = retriever.search(
                retry_query,
                top_k=INITIAL_RETRIEVAL,
                firm=current_firm
                if current_firm and not comparison
                else None,
            )

            retry_results = deduplicate_results(
                retry_results
            )

            # Merge old and new evidence.
            merged = (
                results
                + retry_results
            )

            merged = deduplicate_results(
                merged
            )

            # Sort by retriever score.
            merged.sort(
                key=lambda x: float(
                    x.get("score", 0)
                ),
                reverse=True,
            )

            results = merged[
                :FINAL_EVIDENCE
            ]

    # --------------------------------------------------------
    # If verification repeatedly failed, do NOT pretend.
    # --------------------------------------------------------

    return {
        "answer": (
            "I couldn't verify the proposed answer "
            "against the PFR evidence with sufficient confidence.\n\n"
            "I found these potentially relevant sources:\n\n"
            + "\n".join(
                f"- {r['title']} — {r['url']}"
                for r in results[:3]
            )
        ),
        "current_firm": current_firm,
        "sources": results,
    }


# ============================================================
# MAIN CHAT LOOP
# ============================================================

def main():

    print("=" * 60)
    print("PFR AGENT V3")
    print("=" * 60)

    data_dir = (
        Path(__file__).parent
        / "data"
    )

    try:
        retriever = PFRRetriever(
            data_dir
        )
    except Exception as exc:

        print("\nERROR loading PFR database:")
        print(exc)

        sys.exit(1)

    try:
        client = get_client()
    except Exception as exc:

        print("\nERROR creating MiniMax client:")
        print(exc)

        sys.exit(1)

    model = DEFAULT_MODEL

    print(
        f"Loaded {len(retriever.documents)} searchable chunks"
    )

    print(
        "Firms:",
        ", ".join(
            retriever.firms()
        ),
    )

    if client:
        print(
            f"AI mode: MiniMax ({model})"
        )
    else:
        print(
            "AI mode: MiniMax unavailable"
        )

    print(
        "\nConversation memory: ON"
    )

    print(
        "Smart retrieval: ON"
    )

    print(
        "Evidence verification: ON"
    )

    print(
        "Self-correction: ON"
    )

    print(
        "\nType 'exit' to quit.\n"
    )

    history: list[dict] = []

    current_firm = None

    while True:

        try:
            question = input(
                "You: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye."
            )

            break

        # ====================================================
        # CASUAL CONVERSATION
        # ====================================================

        if (
            is_casual(question)
            and not looks_like_pfr_question(
                question
            )
        ):

            response = casual_answer(
                client=client,
                model=model,
                question=question,
                history=history,
            )

            print(
                "\nPFR Agent:\n"
            )

            print(
                response
            )

            print()

            history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            history = history[
                -MAX_HISTORY_MESSAGES:
            ]

            continue

        # ====================================================
        # PFR QUESTION
        # ====================================================

        try:

            result = answer_pfr(
                question=question,
                retriever=retriever,
                client=client,
                model=model,
                history=history,
                current_firm=current_firm,
            )

            current_firm = result.get(
                "current_firm",
                current_firm,
            )

            response = result[
                "answer"
            ]

            print(
                "\nPFR Agent:\n"
            )

            print(
                response
            )

            print()

            history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            history = history[
                -MAX_HISTORY_MESSAGES:
            ]

        except Exception as exc:

            print(
                "\nERROR:"
            )

            print(
                exc
            )

            print()


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import time
from pathlib import Path

from retriever import PFRRetriever

# We import the real agent engine.
# The benchmark does NOT pass expected answers to it.
from agent import (
    get_client,
    answer_pfr,
    casual_answer,
)


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

MODEL = "MiniMax-M2.7"


# ============================================================
# EVALUATION V2
# ============================================================
#
# IMPORTANT:
#
# The expected answers below are ONLY for the evaluator.
#
# They are NEVER sent to the PFR Agent.
#
# The agent receives only:
#
#     user question
#     conversation history
#     PFR database evidence
#
# The evaluator separately checks the result.
#
# ============================================================


TESTS = [

    # ========================================================
    # LEVEL 1 — DIRECT FACT
    # ========================================================

    {
        "id": "DIRECT_FS_POSITION",
        "type": "pfr",
        "question": (
            "Does FundedSquad allow 3 positions "
            "on the same asset?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "No. FundedSquad prohibits 3 or more open "
            "positions on the same asset, so the maximum "
            "allowed is 2."
        ),
        "required_source": "Position Management Rules",
        "must_contain": [
            "2",
        ],
        "must_not_contain_wrong": [
            "5 positions are allowed",
            "unlimited positions",
        ],
    },


    # ========================================================
    # LEVEL 2 — PARAPHRASE
    # ========================================================

    {
        "id": "PARAPHRASE_FS_POSITION",
        "type": "pfr",
        "question": (
            "I've already got two trades running on one "
            "instrument. Can I open another one at FundedSquad?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "No. Opening a third position would mean 3 open "
            "positions on the same asset, which is prohibited."
        ),
        "required_source": "Position Management Rules",
        "must_contain": [
            "3",
        ],
    },


    # ========================================================
    # LEVEL 3 — TYPO
    # ========================================================

    {
        "id": "TYPO_FS_POSITION",
        "type": "pfr",
        "question": (
            "what is max posision on same asett "
            "in funded squad"
        ),
        "firm": "FundedSquad",
        "expected": (
            "The maximum is 2 open positions on the same asset."
        ),
        "required_source": "Position Management Rules",
        "must_contain": [
            "2",
        ],
    },


    # ========================================================
    # LEVEL 4 — REASONING
    # ========================================================

    {
        "id": "REASONING_FS_POSITION",
        "type": "pfr",
        "question": (
            "If FundedSquad says 3 or more open positions "
            "on one asset are not allowed, what is the "
            "maximum number I can have?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "The maximum is 2 open positions on that asset."
        ),
        "required_source": "Position Management Rules",
        "must_contain": [
            "2",
        ],
    },


    # ========================================================
    # LEVEL 5 — LAYERING
    # ========================================================

    {
        "id": "LAYERING_FS",
        "type": "pfr",
        "question": (
            "Is layering allowed in FundedSquad?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "The answer must be based on FundedSquad's "
            "Position Management Rules and must explain "
            "the restriction on 3 or more positions on "
            "one asset."
        ),
        "required_source": "Position Management Rules",
        "must_contain": [
            "3",
        ],
    },


    # ========================================================
    # LEVEL 6 — RISK
    # ========================================================

    {
        "id": "FS_MAX_RISK",
        "type": "pfr",
        "question": (
            "What is FundedSquad's max risk rule?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "The answer must describe the FundedSquad "
            "Max Risk Rule from the relevant source."
        ),
        "required_source": "Max Risk Rule",
        "must_not_contain_wrong": [
            "FTMO's rule",
            "FundedNext's rule",
        ],
    },


    # ========================================================
    # LEVEL 7 — FLOATING RISK
    # ========================================================

    {
        "id": "FS_FLOATING",
        "type": "pfr",
        "question": (
            "Does FundedSquad have a floating loss rule?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "The answer must distinguish FundedSquad's "
            "floating max risk rule and its applicable "
            "account/stage rather than treating it as "
            "a universal rule for every account."
        ),
        "required_source": "Floating Max Risk Rule",
        "must_not_contain_wrong": [
            "applies to every FundedSquad account",
            "applies to all accounts",
        ],
    },


    # ========================================================
    # LEVEL 8 — FTMO ISOLATION
    # ========================================================

    {
        "id": "FTMO_NEWS",
        "type": "pfr",
        "question": (
            "Does FTMO allow news trading?"
        ),
        "firm": "FTMO",
        "expected": (
            "Answer using FTMO's news-trading source only."
        ),
        "required_source": "Can I trade news?",
        "must_not_contain_wrong": [
            "FundedSquad",
            "FundedNext",
            "FundingPips",
        ],
    },


    # ========================================================
    # LEVEL 9 — FUNDINGPIPS
    # ========================================================

    {
        "id": "FUNDINGPIPS_WEEKEND",
        "type": "pfr",
        "question": (
            "Can I hold trades over the weekend "
            "at FundingPips?"
        ),
        "firm": "FundingPips",
        "expected": (
            "Answer using FundingPips' News Trading & "
            "Weekend Holding source."
        ),
        "required_source": (
            "News Trading & Weekend Holding"
        ),
        "must_not_contain_wrong": [
            "FundedSquad",
            "FTMO",
        ],
    },


    # ========================================================
    # LEVEL 10 — FUNDEDNEXT
    # ========================================================

    {
        "id": "FUNDEDNEXT_WEEKEND",
        "type": "pfr",
        "question": (
            "Does FundedNext allow holding trades "
            "over the weekend?"
        ),
        "firm": "FundedNext",
        "expected": (
            "Answer using FundedNext's own weekend "
            "holding source."
        ),
        "required_source": (
            "Does FundedNext Allow Holding Trades "
            "Over the night & weekend?"
        ),
        "must_not_contain_wrong": [
            "FundedSquad",
            "FTMO",
            "FundingPips",
        ],
    },


    # ========================================================
    # LEVEL 11 — ADVERSARIAL
    # ========================================================

    {
        "id": "ADVERSARIAL_FS",
        "type": "pfr",
        "question": (
            "I heard FundedSquad lets me stack 5 positions "
            "on the same asset. That's correct, right?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "The agent must reject the unsupported 5-position "
            "premise and use the Position Management Rules."
        ),
        "required_source": "Position Management Rules",
        "must_not_contain_wrong": [
            "5 positions are allowed",
            "yes, 5 positions",
            "5 positions are permitted",
        ],
    },


    # ========================================================
    # LEVEL 12 — UNSUPPORTED QUESTION
    # ========================================================

    {
        "id": "UNSUPPORTED_FS",
        "type": "pfr",
        "question": (
            "Does FundedSquad reset the same-asset position "
            "counter at exactly 12:00 UTC every day?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "The agent should not invent a reset time if "
            "the supplied evidence does not establish one."
        ),
        "required_source": None,
        "must_contain_one_of": [
            "couldn't verify",
            "cannot verify",
            "not specified",
            "not stated",
            "no evidence",
            "not found",
        ],
    },


    # ========================================================
    # LEVEL 13 — NORMAL CONVERSATION
    # ========================================================

    {
        "id": "CASUAL_HELLO",
        "type": "casual",
        "question": "How are you?",
        "expected": (
            "Natural conversational response. "
            "It must not talk about PFR evidence."
        ),
        "must_not_contain_wrong": [
            "PFR database",
            "couldn't verify",
            "source evidence",
            "retrieval",
        ],
    },


    # ========================================================
    # LEVEL 14 — NATURAL HUMAN WORDING
    # ========================================================

    {
        "id": "NATURAL_FS",
        "type": "pfr",
        "question": (
            "Bro I've got 2 EURUSD trades open with FundedSquad. "
            "Can I throw one more on?"
        ),
        "firm": "FundedSquad",
        "expected": (
            "No. A third position would reach 3 positions "
            "on the same asset, which is prohibited."
        ),
        "required_source": "Position Management Rules",
        "must_contain": [
            "3",
        ],
    },
]


# ============================================================
# HELPERS
# ============================================================


def contains_any(
    text: str,
    phrases: list[str],
) -> bool:

    text_lower = text.lower()

    return any(
        phrase.lower() in text_lower
        for phrase in phrases
    )


def contains_all(
    text: str,
    phrases: list[str],
) -> bool:

    text_lower = text.lower()

    return all(
        phrase.lower() in text_lower
        for phrase in phrases
    )


def source_is_present(
    results: list[dict],
    required_source: str | None,
) -> bool:

    if not required_source:
        return True

    required = required_source.lower()

    for result in results:

        title = str(
            result.get("title", "")
        ).lower()

        if required in title:
            return True

    return False


# ============================================================
# RUN ONE PFR TEST
# ============================================================


def run_pfr_test(
    retriever,
    client,
    test,
):

    history = []

    current_firm = test.get(
        "firm"
    )

    started = time.perf_counter()

    result = answer_pfr(
        question=test["question"],
        retriever=retriever,
        client=client,
        model=MODEL,
        history=history,
        current_firm=current_firm,
    )

    elapsed = (
        time.perf_counter()
        - started
    ) * 1000

    answer = result["answer"]

    sources = result.get(
        "sources",
        [],
    )

    # --------------------------------------------------------
    # Deterministic checks
    # --------------------------------------------------------

    source_pass = source_is_present(
        sources,
        test.get("required_source"),
    )

    required_pass = True

    if test.get("must_contain"):
        required_pass = contains_all(
            answer,
            test["must_contain"],
        )

    wrong_pass = True

    if test.get("must_not_contain_wrong"):
        wrong_pass = not contains_any(
            answer,
            test["must_not_contain_wrong"],
        )

    one_of_pass = True

    if test.get("must_contain_one_of"):
        one_of_pass = contains_any(
            answer,
            test["must_contain_one_of"],
        )

    return {
        "id": test["id"],
        "type": "pfr",
        "question": test["question"],
        "expected": test["expected"],
        "answer": answer,
        "latency_ms": round(
            elapsed,
            2,
        ),
        "source_pass": source_pass,
        "required_content_pass": required_pass,
        "wrong_content_pass": wrong_pass,
        "unsupported_response_pass": one_of_pass,
        "retrieved_sources": [
            {
                "firm": r.get("firm"),
                "title": r.get("title"),
                "score": r.get("score"),
                "url": r.get("url"),
            }
            for r in sources[:5]
        ],
    }


# ============================================================
# RUN CASUAL TEST
# ============================================================


def run_casual_test(
    client,
    test,
):

    started = time.perf_counter()

    answer = casual_answer(
        client=client,
        model=MODEL,
        question=test["question"],
        history=[],
    )

    elapsed = (
        time.perf_counter()
        - started
    ) * 1000

    wrong_pass = not contains_any(
        answer,
        test.get(
            "must_not_contain_wrong",
            [],
        ),
    )

    return {
        "id": test["id"],
        "type": "casual",
        "question": test["question"],
        "expected": test["expected"],
        "answer": answer,
        "latency_ms": round(
            elapsed,
            2,
        ),
        "casual_behavior_pass": wrong_pass,
    }


# ============================================================
# MULTI-TURN TEST
# ============================================================


def run_multiturn_test(
    retriever,
    client,
):

    print()
    print("=" * 80)
    print("MULTI-TURN CONTEXT TEST")
    print("=" * 80)

    history = []

    current_firm = None

    turns = [
        "Does FundedSquad allow layering?",
        "What about 2 positions?",
        "And what about 3?",
        "Are you sure?",
    ]

    answers = []

    started = time.perf_counter()

    for question in turns:

        result = answer_pfr(
            question=question,
            retriever=retriever,
            client=client,
            model=MODEL,
            history=history,
            current_firm=current_firm,
        )

        answer = result["answer"]

        current_firm = result.get(
            "current_firm",
            current_firm,
        )

        answers.append(
            answer
        )

        history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        print()
        print(f"USER: {question}")
        print()
        print(f"AGENT: {answer}")

    elapsed = (
        time.perf_counter()
        - started
    ) * 1000

    combined = "\n".join(
        answers
    ).lower()

    # We want the conversation to remain on FundedSquad.
    firm_pass = (
        "fundedsquad" in combined
        or "funded squad" in combined
    )

    # We expect the 3-position question to eventually
    # produce the 2-position maximum logic.
    reasoning_pass = (
        "2" in combined
        and "3" in combined
    )

    return {
        "id": "MULTITURN_CONTEXT_001",
        "type": "multiturn",
        "turns": turns,
        "answers": answers,
        "firm_context_pass": firm_pass,
        "reasoning_signal_pass": reasoning_pass,
        "latency_ms": round(
            elapsed,
            2,
        ),
    }


# ============================================================
# PRINT RESULTS
# ============================================================


def print_result(result):

    print()
    print("=" * 80)

    print(
        f"TEST: {result['id']}"
    )

    print(
        f"QUESTION: {result['question']}"
    )

    print(
        f"\nAGENT ANSWER:\n"
        f"{result['answer']}"
    )

    print(
        f"\nLatency: "
        f"{result['latency_ms']} ms"
    )

    if result["type"] == "pfr":

        print(
            "\nSOURCE CHECK: "
            + (
                "PASS"
                if result["source_pass"]
                else "FAIL"
            )
        )

        print(
            "REQUIRED CONTENT CHECK: "
            + (
                "PASS"
                if result["required_content_pass"]
                else "FAIL"
            )
        )

        print(
            "WRONG-CONTENT CHECK: "
            + (
                "PASS"
                if result["wrong_content_pass"]
                else "FAIL"
            )
        )

        print(
            "UNSUPPORTED-RESPONSE CHECK: "
            + (
                "PASS"
                if result["unsupported_response_pass"]
                else "FAIL"
            )
        )

        if result[
            "retrieved_sources"
        ]:

            print(
                "\nSOURCES USED:"
            )

            for source in result[
                "retrieved_sources"
            ]:

                print(
                    f"- {source['firm']} | "
                    f"{source['title']} | "
                    f"{source['score']}"
                )

    elif result["type"] == "casual":

        print(
            "\nCASUAL BEHAVIOR CHECK: "
            + (
                "PASS"
                if result[
                    "casual_behavior_pass"
                ]
                else "FAIL"
            )
        )


# ============================================================
# SUMMARY
# ============================================================


def print_summary(
    results,
    multiturn,
):

    print()
    print()
    print("#" * 80)
    print("PFR AGENT V2 BENCHMARK")
    print("#" * 80)

    pfr_results = [
        r
        for r in results
        if r["type"] == "pfr"
    ]

    casual_results = [
        r
        for r in results
        if r["type"] == "casual"
    ]

    total_pfr = len(
        pfr_results
    )

    source_pass = sum(
        r["source_pass"]
        for r in pfr_results
    )

    content_pass = sum(
        r["required_content_pass"]
        for r in pfr_results
    )

    wrong_pass = sum(
        r["wrong_content_pass"]
        for r in pfr_results
    )

    unsupported_pass = sum(
        r["unsupported_response_pass"]
        for r in pfr_results
    )

    casual_pass = sum(
        r["casual_behavior_pass"]
        for r in casual_results
    )

    print()
    print(
        f"PFR tests: "
        f"{total_pfr}"
    )

    print(
        f"Source grounding: "
        f"{source_pass}/{total_pfr}"
    )

    print(
        f"Required-content checks: "
        f"{content_pass}/{total_pfr}"
    )

    print(
        f"Wrong-content resistance: "
        f"{wrong_pass}/{total_pfr}"
    )

    print(
        f"Unsupported-question behavior: "
        f"{unsupported_pass}/{total_pfr}"
    )

    print(
        f"Casual behavior: "
        f"{casual_pass}/{len(casual_results)}"
    )

    print()
    print(
        "Multi-turn firm context: "
        + (
            "PASS"
            if multiturn[
                "firm_context_pass"
            ]
            else "FAIL"
        )
    )

    print(
        "Multi-turn reasoning signal: "
        + (
            "PASS"
            if multiturn[
                "reasoning_signal_pass"
            ]
            else "FAIL"
        )
    )

    print()

    all_basic_pass = (
        source_pass == total_pfr
        and content_pass == total_pfr
        and wrong_pass == total_pfr
        and unsupported_pass == total_pfr
        and casual_pass == len(
            casual_results
        )
        and multiturn[
            "firm_context_pass"
        ]
        and multiturn[
            "reasoning_signal_pass"
        ]
    )

    if all_basic_pass:

        print(
            "🎯 BASIC V2 BENCHMARK: PASS"
        )

    else:

        print(
            "⚠️ BASIC V2 BENCHMARK: "
            "FAILURES DETECTED"
        )

    print()
    print("#" * 80)


# ============================================================
# SAVE REPORT
# ============================================================


def save_report(
    results,
    multiturn,
):

    report_dir = (
        PROJECT_DIR
        / "evaluation_reports"
    )

    report_dir.mkdir(
        exist_ok=True
    )

    path = (
        report_dir
        / "agent_v2_report.json"
    )

    report = {
        "benchmark": (
            "PFR Agent V2"
        ),
        "model": MODEL,
        "results": results,
        "multiturn": multiturn,
    }

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Full report saved to:\n"
        f"{path}"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 80)
    print("PFR AGENT — FULL SYSTEM BENCHMARK V2")
    print("=" * 80)

    print(
        "\nLoading PFR database..."
    )

    retriever = PFRRetriever(
        DATA_DIR
    )

    print(
        f"Loaded "
        f"{len(retriever.documents)} "
        f"searchable chunks."
    )

    client = get_client()

    if client is None:

        print()
        print(
            "ERROR: MINIMAX_API_KEY was not found."
        )

        print(
            "Make sure your .env file contains:"
        )

        print(
            "MINIMAX_API_KEY=your_key"
        )

        return

    print(
        "\nMiniMax connection: OK"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"\nRunning "
        f"{len(TESTS)} "
        f"system tests..."
    )

    results = []

    for test in TESTS:

        if test["type"] == "pfr":

            result = run_pfr_test(
                retriever,
                client,
                test,
            )

        elif test["type"] == "casual":

            result = run_casual_test(
                client,
                test,
            )

        else:

            continue

        results.append(
            result
        )

        print_result(
            result
        )

    multiturn = run_multiturn_test(
        retriever,
        client,
    )

    print_summary(
        results,
        multiturn,
    )

    save_report(
        results,
        multiturn,
    )


if __name__ == "__main__":
    main()
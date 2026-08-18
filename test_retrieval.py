from retriever import PFRRetriever


def print_results(retriever, question, top_k=10):
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)
    print("=" * 80)

    results = retriever.search(
        question,
        top_k=top_k,
    )

    if not results:
        print("\nNO RESULTS FOUND.")
        return

    print("\nTOP RESULTS:\n")

    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['firm']} | "
            f"{result['title']} | "
            f"score={result['score']}"
        )
        print(f"   {result['url']}")

    print("\n" + "-" * 80)
    print("TOP RESULT CONTENT:")
    print("-" * 80)

    top = results[0]

    print(f"Firm: {top['firm']}")
    print(f"Title: {top['title']}")
    print(f"URL: {top['url']}")
    print(f"Score: {top['score']}")
    print("\nCONTENT:\n")
    print(top["content"])

    print("\n" + "=" * 80)


def main():
    retriever = PFRRetriever("data")

    print("=" * 80)
    print("PFR RETRIEVAL TEST")
    print("=" * 80)

    print(
        f"\nLoaded {len(retriever.documents)} searchable chunks"
    )

    print(
        "Firms:",
        ", ".join(retriever.firms())
    )

    # Original tests
    questions = [
        "Does FTMO allow news trading?",

        "Can I hold trades over the weekend at FundingPips?",

        "What is FundedSquad's max risk rule?",

        "What is the performance fee rule at Alpha Capital Group?",

        "Does FundedNext allow holding trades over the weekend?",

        # IMPORTANT:
        # This is the exact question that exposed our retrieval problem.
        "is there any max position holding on the same asset in Funded Squad",

        # Typo test.
        # This checks whether the new fuzzy retrieval can understand
        # "posision" instead of "position".
        "is there any max posision holding on the same asset in Funded Squad",

        # Concept test.
        "Can I have 3 open trades on the same asset in FundedSquad?",

        # Layering terminology test.
        "Is layering allowed in FundedSquad?",

        # Follow-up-style retrieval test.
        "What is the maximum number of positions on one asset?",
    ]

    for question in questions:
        print_results(
            retriever,
            question,
            top_k=10,
        )


if __name__ == "__main__":
    main()
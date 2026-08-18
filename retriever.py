from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "can", "could", "do", "does", "for", "from", "how", "i",
    "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "should", "the", "their", "there", "this", "to",
    "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "about", "allow", "allowed",
    "am", "during", "over", "trade", "trading", "account",
    "accounts", "please", "tell", "give", "any", "there",
    "than", "then", "same"
}


# ============================================================
# PROP-FIRM ALIASES
# ============================================================

FIRM_ALIASES = {
    "fundingpips": [
        "fundingpips",
        "funding pips",
    ],
    "fundednext": [
        "fundednext",
        "funded next",
    ],
    "ftmo": [
        "ftmo",
    ],
    "fundedsquad": [
        "fundedsquad",
        "funded squad",
    ],
    "alpha_capital_group": [
        "alpha capital group",
        "alphacapitalgroup",
        "alpha capital",
    ],
}


FIRM_CANONICAL = {
    "fundingpips": "FundingPips",
    "fundednext": "FundedNext",
    "ftmo": "FTMO",
    "fundedsquad": "FundedSquad",
    "alpha_capital_group": "Alpha Capital Group",
}


# ============================================================
# QUERY / CONCEPT ALIASES
#
# These are deliberately conservative.
# They improve retrieval without changing the actual answer.
# ============================================================

CONCEPT_ALIASES = {

    # Position quantity / layering
    "position": [
        "position",
        "positions",
        "open position",
        "open positions",
        "position limit",
        "maximum positions",
        "max positions",
        "number of positions",
        "position management",
        "layering",
    ],

    "positions": [
        "position",
        "positions",
        "open positions",
        "position limit",
        "maximum positions",
        "max positions",
        "position management",
        "layering",
    ],

    "posision": [
        "position",
        "positions",
        "open positions",
        "position limit",
        "maximum positions",
        "max positions",
        "position management",
        "layering",
    ],

    "posistion": [
        "position",
        "positions",
        "open positions",
        "position limit",
        "maximum positions",
        "max positions",
        "position management",
        "layering",
    ],

    "pos": [
        "position",
        "positions",
        "open positions",
    ],

    # Asset / instrument
    "asset": [
        "asset",
        "single asset",
        "same asset",
        "instrument",
        "same instrument",
        "trading instrument",
    ],

    "instrument": [
        "instrument",
        "asset",
        "same asset",
        "same instrument",
    ],

    # Maximum / limit
    "max": [
        "maximum",
        "max",
        "limit",
        "allowed",
        "restriction",
    ],

    "maximum": [
        "maximum",
        "max",
        "limit",
        "allowed",
        "restriction",
    ],

    "limit": [
        "limit",
        "maximum",
        "max",
        "restriction",
        "allowed",
    ],

    # Layering
    "layer": [
        "layering",
        "positions",
        "open positions",
        "same asset",
        "same instrument",
    ],

    "layering": [
        "layering",
        "positions",
        "open positions",
        "same asset",
        "same instrument",
    ],

    # Floating PNL / floating loss
    "floating": [
        "floating",
        "floating pnl",
        "floating loss",
        "unrealized loss",
        "open position loss",
    ],

    "pnl": [
        "pnl",
        "floating pnl",
        "floating loss",
        "profit loss",
        "unrealized loss",
    ],

    # Drawdown
    "drawdown": [
        "drawdown",
        "maximum loss",
        "max loss",
        "daily loss",
        "overall loss",
    ],

    # Weekend
    "weekend": [
        "weekend",
        "weekend holding",
        "hold over weekend",
        "holding over weekend",
        "saturday",
        "sunday",
    ],

    # News
    "news": [
        "news trading",
        "news",
        "economic news",
        "high impact news",
        "news events",
    ],

    # Payout
    "payout": [
        "payout",
        "withdrawal",
        "withdraw",
        "profit split",
        "performance fee",
    ],

    # Allocation
    "allocation": [
        "allocation",
        "maximum allocation",
        "max allocation",
        "funding allocation",
        "capital allocation",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text: str) -> str:
    text = (text or "").lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9$%:+\-./ ]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def tokens(text: str) -> list[str]:
    words = re.findall(
        r"[a-z0-9]+(?:[%$.:+\-][a-z0-9]+)*",
        normalize(text),
    )

    return [
        w
        for w in words
        if w not in STOPWORDS and len(w) > 1
    ]


# ============================================================
# CHUNKING
# ============================================================

def split_into_chunks(
    text: str,
    max_words: int = 260,
    overlap: int = 45,
) -> list[str]:

    words = (text or "").split()

    if len(words) <= max_words:
        return [
            text.strip()
        ] if text.strip() else []

    chunks = []

    step = max_words - overlap

    for start in range(
        0,
        len(words),
        step,
    ):
        chunk = " ".join(
            words[
                start:start + max_words
            ]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if start + max_words >= len(words):
            break

    return chunks


# ============================================================
# FUZZY WORD SIMILARITY
# ============================================================

def fuzzy_similarity(
    a: str,
    b: str,
) -> float:

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Strong protection against very short accidental matches.
    if len(a) < 4 or len(b) < 4:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def best_fuzzy_match(
    query_term: str,
    document_terms: set[str],
) -> tuple[str | None, float]:

    best_term = None
    best_score = 0.0

    for candidate in document_terms:

        similarity = fuzzy_similarity(
            query_term,
            candidate,
        )

        if similarity > best_score:
            best_score = similarity
            best_term = candidate

    return best_term, best_score


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query: str) -> list[str]:

    normalized = normalize(query)

    base_tokens = tokens(normalized)

    expanded = list(base_tokens)

    # --------------------------------------------------------
    # Add aliases for concepts.
    # --------------------------------------------------------

    for term in base_tokens:

        aliases = CONCEPT_ALIASES.get(
            term,
            [],
        )

        expanded.extend(
            aliases
        )

    # --------------------------------------------------------
    # Detect useful multi-word concepts.
    # --------------------------------------------------------

    concept_phrases = [

        (
            ["max", "position"],
            [
                "maximum positions",
                "max positions",
                "open positions",
                "position limit",
                "position management",
                "layering",
            ],
        ),

        (
            ["maximum", "position"],
            [
                "maximum positions",
                "open positions",
                "position limit",
                "position management",
                "layering",
            ],
        ),

        (
            ["same", "asset"],
            [
                "same asset",
                "single asset",
                "same instrument",
                "single instrument",
                "open positions",
                "layering",
            ],
        ),

        (
            ["open", "position"],
            [
                "open positions",
                "position limit",
                "position management",
                "layering",
            ],
        ),

        (
            ["floating", "pnl"],
            [
                "floating pnl",
                "floating loss",
                "unrealized loss",
                "max risk",
            ],
        ),

        (
            ["weekend", "holding"],
            [
                "weekend holding",
                "hold over weekend",
                "holding over weekend",
            ],
        ),

        (
            ["news", "trading"],
            [
                "news trading",
                "news events",
                "economic news",
            ],
        ),
    ]

    for required_terms, additions in concept_phrases:

        if all(
            term in base_tokens
            for term in required_terms
        ):
            expanded.extend(additions)

    # --------------------------------------------------------
    # Remove duplicates while preserving order.
    # --------------------------------------------------------

    final = []

    seen = set()

    for item in expanded:

        item_norm = normalize(item)

        if not item_norm:
            continue

        if item_norm in seen:
            continue

        seen.add(item_norm)
        final.append(item_norm)

    return final


# ============================================================
# RETRIEVER
# ============================================================

class PFRRetriever:

    def __init__(
        self,
        data_dir: str | Path = "data",
    ):

        self.data_dir = Path(
            data_dir
        )

        self.documents: list[
            dict[str, Any]
        ] = []

        self.df = Counter()

        self.avgdl = 0.0

        self._load()

    # ========================================================
    # FIRM DETECTION
    # ========================================================

    def _infer_firm(
        self,
        filename: str,
    ) -> str:

        name = filename.lower()

        if "fundingpips" in name:
            return "FundingPips"

        if "fundednext" in name:
            return "FundedNext"

        if "fundedsquad" in name:
            return "FundedSquad"

        if "ftmo" in name:
            return "FTMO"

        if "alpha" in name:
            return "Alpha Capital Group"

        return Path(filename).stem

    # ========================================================
    # LOAD JSON DATABASE
    # ========================================================

    def _load(self):

        files = sorted(
            self.data_dir.glob("*.json")
        )

        for path in files:

            if path.name == "manifest.json":
                continue

            try:

                raw = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception as exc:

                print(
                    f"Warning: could not read {path}: {exc}"
                )

                continue

            if isinstance(raw, dict):
                raw = [raw]

            firm = self._infer_firm(
                path.name
            )

            for article in raw:

                title = str(
                    article.get(
                        "title",
                        "",
                    )
                    or ""
                )

                url = str(
                    article.get(
                        "url",
                        "",
                    )
                    or ""
                )

                category = str(
                    article.get(
                        "category",
                        "",
                    )
                    or ""
                )

                section = str(
                    article.get(
                        "section",
                        "",
                    )
                    or ""
                )

                content = str(
                    article.get(
                        "content",
                        "",
                    )
                    or ""
                )

                if not content.strip():
                    continue

                for chunk_no, chunk in enumerate(
                    split_into_chunks(
                        content
                    )
                ):

                    searchable = (
                        f"{firm} "
                        f"{title} "
                        f"{category} "
                        f"{section} "
                        f"{chunk}"
                    )

                    toks = tokens(
                        searchable
                    )

                    if not toks:
                        continue

                    doc = {
                        "id": (
                            f"{path.stem}:"
                            f"{len(self.documents)}"
                        ),
                        "firm": firm,
                        "title": title,
                        "url": url,
                        "category": (
                            article.get(
                                "category"
                            )
                        ),
                        "section": (
                            article.get(
                                "section"
                            )
                        ),
                        "content": chunk,
                        "tokens": toks,
                        "chunk": chunk_no,
                    }

                    self.documents.append(
                        doc
                    )

                    for term in set(toks):

                        self.df[term] += 1

        total_len = sum(
            len(d["tokens"])
            for d in self.documents
        )

        self.avgdl = (
            total_len
            / max(
                1,
                len(self.documents),
            )
        )

    # ========================================================
    # PUBLIC HELPERS
    # ========================================================

    def firms(self) -> list[str]:

        return sorted(
            {
                d["firm"]
                for d in self.documents
            }
        )

    # ========================================================
    # FIRM DETECTION
    # ========================================================

    def _detect_firm(
        self,
        query: str,
    ) -> str | None:

        q = normalize(query)

        # Longer aliases first.
        aliases = []

        for firm, names in FIRM_ALIASES.items():

            for alias in names:

                aliases.append(
                    (
                        len(alias),
                        alias,
                        firm,
                    )
                )

        aliases.sort(
            reverse=True
        )

        for _, alias, firm in aliases:

            if alias in q:

                return FIRM_CANONICAL[
                    firm
                ]

        return None

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 8,
        firm: str | None = None,
    ) -> list[dict[str, Any]]:

        q_norm = normalize(
            query
        )

        if not q_norm:
            return []

        base_tokens = tokens(
            q_norm
        )

        if not base_tokens:
            return []

        # ----------------------------------------------------
        # Detect firm.
        # ----------------------------------------------------

        detected = (
            firm
            or self._detect_firm(
                query
            )
        )

        # ----------------------------------------------------
        # Firm filtering.
        # ----------------------------------------------------

        if detected:

            allowed = [
                d
                for d in self.documents
                if d["firm"].lower()
                == detected.lower()
            ]

            if not allowed:
                allowed = self.documents

        else:

            allowed = self.documents

        # ----------------------------------------------------
        # Expand query.
        # ----------------------------------------------------

        expanded_terms = expand_query(
            q_norm
        )

        expanded_tokens = []

        for item in expanded_terms:

            expanded_tokens.extend(
                tokens(item)
            )

        # Remove duplicates.
        expanded_tokens = list(
            dict.fromkeys(
                expanded_tokens
            )
        )

        # ----------------------------------------------------
        # BM25 parameters.
        # ----------------------------------------------------

        N = max(
            1,
            len(self.documents),
        )

        k1 = 1.5
        b = 0.75

        scored = []

        # ====================================================
        # SCORE EVERY CANDIDATE
        # ====================================================

        for d in allowed:

            doc_tokens = d["tokens"]

            tf = Counter(
                doc_tokens
            )

            dl = len(
                doc_tokens
            )

            doc_token_set = set(
                doc_tokens
            )

            title_norm = normalize(
                d["title"]
            )

            title_tokens = set(
                tokens(
                    d["title"]
                )
            )

            content_norm = normalize(
                d["content"]
            )

            score = 0.0

            # ------------------------------------------------
            # 1. Original query BM25.
            # ------------------------------------------------

            for term in base_tokens:

                if term not in tf:
                    continue

                df = self.df.get(
                    term,
                    0,
                )

                idf = math.log(
                    1
                    + (
                        N
                        - df
                        + 0.5
                    )
                    / (
                        df
                        + 0.5
                    )
                )

                freq = tf[term]

                score += (
                    idf
                    * (
                        (
                            freq
                            * (k1 + 1)
                        )
                        / (
                            freq
                            + k1
                            * (
                                1
                                - b
                                + b
                                * dl
                                / max(
                                    1,
                                    self.avgdl,
                                )
                            )
                        )
                    )
                )

            # ------------------------------------------------
            # 2. Expanded concept matching.
            #
            # This is extremely important for:
            # "max posision same asset"
            # -> position / layering / open positions
            # ------------------------------------------------

            expansion_hits = 0

            for term in expanded_tokens:

                if term in doc_token_set:

                    expansion_hits += 1

                    # Expanded concepts are useful,
                    # but weaker than exact matches.
                    score += 1.8

            # ------------------------------------------------
            # 3. Exact title token matches.
            # ------------------------------------------------

            exact_title_hits = (
                len(
                    set(base_tokens)
                    & title_tokens
                )
            )

            score += (
                8.0
                * exact_title_hits
            )

            # ------------------------------------------------
            # 4. Expanded title matches.
            # ------------------------------------------------

            expanded_title_hits = (
                len(
                    set(expanded_tokens)
                    & title_tokens
                )
            )

            score += (
                7.0
                * expanded_title_hits
            )

            # ------------------------------------------------
            # 5. Fuzzy matching.
            #
            # Handles:
            # posision -> position
            # positon  -> position
            # positons -> positions
            # etc.
            # ------------------------------------------------

            fuzzy_hits = 0

            # Only fuzzy-match meaningful query words.
            for qterm in base_tokens:

                if qterm in doc_token_set:
                    continue

                best_term, similarity = (
                    best_fuzzy_match(
                        qterm,
                        doc_token_set,
                    )
                )

                if (
                    best_term
                    and similarity >= 0.82
                ):

                    fuzzy_hits += 1

                    score += (
                        7.5
                        * similarity
                    )

            # ------------------------------------------------
            # 6. Fuzzy title matching.
            # ------------------------------------------------

            for qterm in base_tokens:

                best_title_term, similarity = (
                    best_fuzzy_match(
                        qterm,
                        title_tokens,
                    )
                )

                if (
                    best_title_term
                    and similarity >= 0.78
                ):

                    score += (
                        12.0
                        * similarity
                    )

            # ------------------------------------------------
            # 7. Important phrase bonuses.
            # ------------------------------------------------

            important_phrases = [
                (
                    "same asset",
                    25.0,
                ),
                (
                    "single asset",
                    25.0,
                ),
                (
                    "same instrument",
                    25.0,
                ),
                (
                    "open positions",
                    30.0,
                ),
                (
                    "position management",
                    35.0,
                ),
                (
                    "maximum positions",
                    35.0,
                ),
                (
                    "max positions",
                    35.0,
                ),
                (
                    "position limit",
                    35.0,
                ),
                (
                    "floating pnl",
                    35.0,
                ),
                (
                    "floating loss",
                    30.0,
                ),
                (
                    "weekend holding",
                    30.0,
                ),
                (
                    "news trading",
                    30.0,
                ),
                (
                    "performance fee",
                    30.0,
                ),
                (
                    "maximum allocation",
                    30.0,
                ),
            ]

            for phrase, bonus in important_phrases:

                if phrase in q_norm:

                    if phrase in title_norm:

                        score += (
                            bonus
                            * 2.5
                        )

                    elif phrase in content_norm:

                        score += bonus

            # ------------------------------------------------
            # 8. Semantic-ish position concept boost.
            #
            # If the question talks about positions on an
            # asset and the article talks about layering or
            # position management, strongly prioritize it.
            # ------------------------------------------------

            position_question = (
                any(
                    term in base_tokens
                    for term in [
                        "position",
                        "positions",
                        "posision",
                        "posistion",
                        "layer",
                        "layering",
                    ]
                )
                or "same asset" in q_norm
                or "open position" in q_norm
            )

            if position_question:

                position_concepts = [
                    "position",
                    "positions",
                    "layering",
                    "layer",
                    "asset",
                    "instrument",
                    "open",
                ]

                concept_hits = sum(
                    1
                    for term
                    in position_concepts
                    if term
                    in doc_token_set
                )

                score += (
                    5.0
                    * concept_hits
                )

                if (
                    "position management"
                    in title_norm
                ):

                    score += 80.0

                if (
                    "layering"
                    in title_norm
                ):

                    score += 70.0

            # ------------------------------------------------
            # 9. Firm boost.
            # ------------------------------------------------

            if (
                detected
                and d["firm"].lower()
                == detected.lower()
            ):

                score += 15.0

            # ------------------------------------------------
            # 10. Exact full query.
            # ------------------------------------------------

            if (
                len(q_norm.split())
                >= 2
                and q_norm
                in content_norm
            ):

                score += 20.0

            # ------------------------------------------------
            # Keep positive results.
            # ------------------------------------------------

            if score > 0:

                scored.append(
                    (
                        score,
                        d,
                    )
                )

        # ====================================================
        # SORT
        # ====================================================

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        # ====================================================
        # RETURN DISTINCT SOURCE PAGES
        # ====================================================

        results = []

        seen_urls = set()

        for score, d in scored:

            url = d.get(
                "url",
                "",
            )

            # Prefer unique source pages.
            if url in seen_urls:
                continue

            seen_urls.add(
                url
            )

            item = dict(d)

            item.pop(
                "tokens",
                None,
            )

            item["score"] = round(
                score,
                4,
            )

            results.append(
                item
            )

            if len(results) >= top_k:
                break

        return results
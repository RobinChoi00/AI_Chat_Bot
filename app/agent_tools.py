"""
Agent Tool implementations.

Each tool is a Python function that the LLM can invoke via OpenAI function-calling.
Tools return strings (formatted text) or dicts (structured) that the LLM uses to
craft the final user-facing response.

Design principles:
- Tools are deterministic and side-effect-aware (e.g. capture_sales_lead spawns email).
- Tools return concise, factual data only — no marketing copy. The LLM phrases it.
- Hybrid retrieval (BM25 + dense) is used for spec/recommendation tools to give
  reliable matches when the user mentions a specific model name.
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# Read lock for FAISS. Injected by main.py during boot — falls back to a
# no-op context manager so the module remains importable standalone (tests).
_faiss_read_lock_factory = None  # type: ignore[assignment]


def set_faiss_read_lock_factory(factory) -> None:
    """Wire up a callable that returns the read-lock context manager.

    main.py calls this with `faiss_rwlock.read` so that every BM25 + dense
    search performed by HybridRetriever blocks while a webhook is rebuilding
    the index. Reads remain concurrent with each other.
    """
    global _faiss_read_lock_factory
    _faiss_read_lock_factory = factory


def _read_lock_ctx():
    if _faiss_read_lock_factory is None:
        return nullcontext()
    return _faiss_read_lock_factory()

# ============================================================================
# Hybrid Retrieval (BM25 + Dense)
# ============================================================================

class HybridRetriever:
    """
    Combines BM25 keyword search with FAISS dense similarity using
    Reciprocal Rank Fusion (RRF). Greatly improves recall when users
    type exact model names like 'Hypnos 4D' or 'Nova II'.
    """

    def __init__(self, vectorstore, bm25_corpus_docs: List[Document]):
        self.vectorstore = vectorstore
        self.docs = bm25_corpus_docs
        try:
            from rank_bm25 import BM25Okapi
            self._tokenized = [self._tokenize(d.page_content) for d in self.docs]
            self.bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        except ImportError:
            logger.warning("rank_bm25 not installed → falling back to dense-only search")
            self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def search(self, query: str, k: int = 8, model_hint: Optional[str] = None) -> List[Document]:
        """
        Returns top-k documents using RRF of BM25 and dense rankings.
        If `model_hint` is provided, documents whose metadata.title matches
        get a strong boost.
        """
        if not query.strip():
            return []

        # Acquire the FAISS read lock for the duration of this search so that
        # an in-flight webhook update doesn't swap docstore contents mid-read.
        with _read_lock_ctx():
            try:
                dense_hits = self.vectorstore.similarity_search(query, k=k * 2)
            except Exception as e:
                logger.warning(f"dense search failed: {e}")
                dense_hits = []

            bm25_hits: List[Document] = []
            if self.bm25 is not None:
                try:
                    tokens = self._tokenize(query)
                    scores = self.bm25.get_scores(tokens)
                    ranked_idx = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[: k * 2]
                    bm25_hits = [self.docs[i] for i in ranked_idx if scores[i] > 0]
                except Exception as e:
                    logger.warning(f"bm25 search failed: {e}")

        # RRF fusion
        rrf_k = 60  # standard
        scored: Dict[str, Tuple[float, Document]] = {}
        for rank, doc in enumerate(dense_hits):
            key = doc.page_content[:200]
            scored.setdefault(key, (0.0, doc))
            scored[key] = (scored[key][0] + 1.0 / (rrf_k + rank + 1), doc)
        for rank, doc in enumerate(bm25_hits):
            key = doc.page_content[:200]
            scored.setdefault(key, (0.0, doc))
            scored[key] = (scored[key][0] + 1.0 / (rrf_k + rank + 1), doc)

        # Model hint boost
        if model_hint:
            hint_norm = re.sub(r"[^a-z0-9]", "", model_hint.lower())
            for key, (score, doc) in list(scored.items()):
                title = (doc.metadata.get("title") or "").lower()
                title_norm = re.sub(r"[^a-z0-9]", "", title)
                if hint_norm and (hint_norm in title_norm or title_norm in hint_norm):
                    scored[key] = (score + 1.0, doc)  # large boost

        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:k]]


# ============================================================================
# Tool implementations (pure functions; Agent calls them)
# ============================================================================

def _extract_price(text: str) -> Optional[float]:
    """Extract the most likely actual chair price from a doc.

    Strategy:
    1. Prefer 'Variant Price: $X' field if present (Shopify export schema).
    2. Otherwise scan all $-prefixed numbers and pick the MAX in a reasonable
       chair price band ($500-$50,000). Picking the min was wrong because docs
       often contain monthly payment ($199/mo), warranty add-on ($99), or
       shipping fee ($249) — all of which are far below the real chair price.
    """
    if not text:
        return None

    variant_match = re.search(r"Variant\s+Price[^\n$]*\$?(\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    if variant_match:
        try:
            v = float(variant_match.group(1).replace(",", ""))
            if 500 <= v <= 50000:
                return v
        except ValueError:
            pass

    # Fall back: pick the MAX $-prefixed price within the chair range.
    matches = re.findall(r"\$\s*(\d[\d,]*\.?\d*)", text)
    prices = []
    for m in matches:
        try:
            v = float(m.replace(",", ""))
            if 500 <= v <= 50000:
                prices.append(v)
        except ValueError:
            continue
    return max(prices) if prices else None


def _extract_title(doc: Document) -> str:
    return doc.metadata.get("title") or doc.page_content.split("\n", 1)[0][:80]


# Keywords that map a user's query topic to matching spec column patterns
_SPEC_TOPIC_KEYWORDS: List[Tuple[List[str], List[str]]] = [
    # (user query keywords, spec column keywords)
    (["door", "doorway", "entrance", "minimum door", "minimum entrance"], ["minimum doorway", "door width"]),
    (["width", "wide"], ["dimension - standing - width", "standing - width"]),
    (["weight", "how heavy", "lbs", "kg"], ["chair weight", "user weight", "maximum user weight"]),
    (["height", "tall", "how tall"], ["height"]),
    (["length", "depth", "how deep", "how long"], ["length"]),
    (["footrest", "foot extension"], ["footrest extension"]),
    (["shoulder"], ["shoulder width"]),
    (["seat"], ["seat width"]),
    (["price", "cost", "how much"], ["variant price", "price"]),
    (["track", "sl track", "sl-track"], ["track - type", "track type"]),
    (["air", "airbag", "air cell"], ["air cell", "airbag"]),
    (["zero gravity", "0 gravity"], ["zero gravity"]),
    (["space saving", "space-saving"], ["space saving"]),
    (["heat", "heating"], ["heat", "heated"]),
    (["warranty"], ["warranty"]),
    (["auto program", "auto programs"], ["auto"]),
    (["manual mode", "manual program"], ["manual"]),
]


def _find_authoritative_lines(doc_content: str, query: str) -> List[str]:
    """
    Extract the exact spec lines that directly answer the query topic.
    Returns a list of matching lines from the document.
    """
    query_lower = query.lower()
    target_col_keywords: List[str] = []

    for user_kws, col_kws in _SPEC_TOPIC_KEYWORDS:
        if any(kw in query_lower for kw in user_kws):
            target_col_keywords.extend(col_kws)

    if not target_col_keywords:
        return []

    hits = []
    for line in doc_content.split("\n"):
        line_lower = line.lower()
        if line.startswith("- ") and any(kw in line_lower for kw in target_col_keywords):
            hits.append(line.strip())
    return hits


# Common user-side spellings / typos / OCR noise that should map to the
# canonical brand or model strings used in the spec sheet.
_MODEL_ALIASES: List[Tuple[str, str]] = [
    (r"\bosaka\b", "osaki"),         # "Osaka" → "Osaki"
    (r"\bosakii\b", "osaki"),        # double-i typo
    (r"\botomic\b", "otamic"),       # "Otomic" → "Otamic"
    (r"\bottomic\b", "otamic"),
    (r"\bottoman\b", "otamic"),
    (r"\botamic\s+le\b", "otamic le"),
    (r"\bhipnos\b", "hypnos"),       # "Hipnos" → "Hypnos"
    (r"\bsojo\b", "soho"),           # "Sojo" → "Soho"
    (r"\bxrest\b", "xrest"),
    # OCR confusion: digit '0' instead of letter 'O' in product codes
    # e.g. "09-4000cs" → "os-4000cs"
    (r"\b0(\d)-?", r"os-\1"),
    (r"\b0s-", "os-"),
    (r"\bo5-", "os-"),
]


def _normalize_model_query(text: str) -> str:
    """Apply common typo / OCR fixes to user-provided model names."""
    if not text:
        return text
    t = text
    for pat, repl in _MODEL_ALIASES:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    # Collapse multiple spaces / dashes
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tool_search_chair_specs(
    *,
    products_retriever: HybridRetriever,
    query: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Look up spec details for one or more chairs. Use when the user asks about
    dimensions, weight, door size, features, price etc. of a specific model.

    Resilience layers (in order):
      1. Exact query with model_name as hint
      2. Typo-normalized query (Osaka→Osaki, 09-→OS-, etc.)
      3. Looser search using only the model name (drop the spec topic)
      4. Final fallback: searching just on the alphanumeric core of the model
    """
    norm_model = _normalize_model_query(model_name) if model_name else None
    base_query = f"{model_name} {query}".strip() if model_name else query

    docs = products_retriever.search(base_query, k=5, model_hint=model_name)

    # Try typo-normalized form if the original returned nothing useful
    if not docs and norm_model and norm_model.lower() != (model_name or "").lower():
        retry_query = f"{norm_model} {query}".strip()
        logger.info(f"[search_chair_specs] retrying with normalized: {retry_query!r}")
        docs = products_retriever.search(retry_query, k=5, model_hint=norm_model)

    # Try just the model name (drop the spec topic) — sometimes the topic noise
    # hurts retrieval (e.g. "parts" doesn't help find a spec sheet)
    if not docs and (model_name or norm_model):
        bare = norm_model or model_name
        logger.info(f"[search_chair_specs] retrying with bare model: {bare!r}")
        docs = products_retriever.search(bare or "", k=5, model_hint=bare or "")

    # Final fallback: pull out alphanumeric core (e.g. "4000cs", "soho", "otamic")
    if not docs and (model_name or norm_model):
        core_tokens = re.findall(r"[a-zA-Z]{3,}|\d{3,}", (norm_model or model_name or ""))
        if core_tokens:
            core_query = " ".join(core_tokens)
            logger.info(f"[search_chair_specs] retrying with core tokens: {core_query!r}")
            docs = products_retriever.search(core_query, k=5, model_hint=core_query)

    if not docs:
        return (
            "NO_RESULTS: No matching specifications found in the catalog. "
            "Possible reasons: (a) the model is an older/discontinued unit, "
            "(b) the spelling differs from our records. "
            "Suggest asking the customer to share the exact model name from "
            "the chair's serial-number sticker, and offer to escalate to support."
        )

    lines: List[str] = []

    # 💰 Token-budget aware: when the query has a clear spec topic and we
    # already found AUTHORITATIVE lines, we no longer dump the full 1200-char
    # body too — the LLM only needs the authoritative numbers to answer.
    # Only the first doc gets the "full context" dump as a safety net.
    for i, doc in enumerate(docs, 1):
        title = _extract_title(doc)
        auth_lines = _find_authoritative_lines(doc.page_content, query)

        lines.append(f"\n--- Result {i}: {title} ---")
        if auth_lines:
            lines.append("AUTHORITATIVE SPEC VALUES (use EXACTLY these numbers, do not paraphrase):")
            for al in auth_lines:
                lines.append(f"  {al}")
            if i == 1:
                # Single, smaller context dump for the top hit only.
                lines.append("\nAdditional context:")
                lines.append(doc.page_content[:600])
        else:
            # No targeted match → include a trimmed body so the LLM can still
            # answer general questions about the model.
            lines.append(doc.page_content[:1000])

    return "\n".join(lines)


def tool_recommend_chairs(
    *,
    products_retriever: HybridRetriever,
    user_need: str,
    budget_min: Optional[float] = None,
    budget_max: Optional[float] = None,
    exclude_models: Optional[List[str]] = None,
    num_recommendations: int = 3,
) -> str:
    """
    Recommend premium massage chairs based on user need. Filters out accessories
    and applies budget constraints if provided.

    Pricing safety nets:
    - Anything < $500 is treated as a non-chair / accessory (a 4D chair below
      $500 is structurally impossible — those came from misparsed lines like
      monthly-payment promos or shipping fees).
    - Budget bands are widened slightly so e.g. "around $3000" returns chairs
      from $2000-$5000 instead of nothing.
    """
    NON_CHAIR_KEYWORDS = {
        "mat", "pad", "cover", "cleaner", "gun", "cushion", "shawl",
        "module", "fragrance", "scraping", "foot spa", "knee", "neck massager",
        "hand massager", "eye massager", "gua sha", "tens", "vending",
        "swivel", "caddo", "bundle", "patio", "zena", "office chair",
        "back seat", "soaking spa", "trainer", "j5 jade",
    }
    MIN_REASONABLE_PRICE = 500.0  # anything below this is not a real chair

    # Slightly widen the band so "around $3000" still returns chairs in the
    # $2000-$5000 range rather than failing entirely.
    eff_min = budget_min
    eff_max = budget_max
    if budget_max is not None and budget_min is None:
        eff_min = max(MIN_REASONABLE_PRICE, budget_max * 0.6)
        eff_max = budget_max * 1.3

    docs = products_retriever.search(user_need or "premium 4D massage chair", k=20)
    candidates = []
    seen_titles = set()
    excludes_norm = {(m or "").lower().strip() for m in (exclude_models or [])}

    for doc in docs:
        title = _extract_title(doc)
        title_lower = title.lower()
        if title_lower in seen_titles:
            continue
        seen_titles.add(title_lower)
        if any(k in title_lower for k in NON_CHAIR_KEYWORDS):
            continue
        if any(ex and ex in title_lower for ex in excludes_norm):
            continue
        price = _extract_price(doc.page_content)
        if price is None or price < MIN_REASONABLE_PRICE:
            continue
        if eff_min is not None and price < eff_min:
            continue
        if eff_max is not None and price > eff_max:
            continue
        candidates.append((title, price, doc.page_content[:500]))

    if not candidates:
        return (
            "NO_RESULTS: No chairs match the given criteria. "
            "Suggest broadening budget (premium massage chairs typically range $1,500-$10,000)."
        )

    # If user gave a target budget, sort by closeness to budget midpoint.
    # Otherwise sort by price descending (premium models first).
    if budget_max is not None:
        target = ((budget_min or budget_max * 0.7) + budget_max) / 2
        candidates.sort(key=lambda x: abs(x[1] - target))
    else:
        candidates.sort(key=lambda x: -x[1])
    picks = candidates[:num_recommendations]

    header = f"Top {len(picks)} matching chairs"
    if budget_max is not None:
        header += f" (target budget ~${budget_max:,.0f})"
    header += ":"
    lines = [header]
    for i, (title, price, body) in enumerate(picks, 1):
        lines.append(f"\n--- Pick {i}: {title} (${price:,.0f}) ---\n{body}")
    lines.append(
        "\nIMPORTANT: Quote ONLY the prices shown above. Do NOT recall prices from training data."
    )
    return "\n".join(lines)


def tool_get_repair_help(
    *,
    qa_retriever: HybridRetriever,
    issue_description: str,
    error_code: Optional[str] = None,
) -> str:
    """Look up repair / error code information."""
    query = f"error code {error_code} {issue_description}" if error_code else issue_description
    docs = qa_retriever.search(query, k=5)
    suppress = "\nFOOTER_HINT: SUPPRESS_LEAD_FOOTER (this is a service issue, do not pitch sales)."
    if not docs:
        return "NO_RESULTS: No specific repair guide found." + suppress
    lines = ["Relevant repair / error info:"]
    for i, doc in enumerate(docs, 1):
        lines.append(f"\n[{i}] {doc.page_content[:800]}")
    lines.append(suppress)
    return "\n".join(lines)


# Topics that are clearly post-purchase / service-oriented — never pitch sales after.
_POLICY_SERVICE_TOPICS = (
    "warranty", "return", "refund", "exchange", "repair", "service",
    "broken", "damaged", "defect", "claim",
)


def tool_get_warranty_or_policy(
    *,
    web_retriever: HybridRetriever,
    topic: str,
) -> str:
    """Retrieve warranty / shipping / return / installation policy info."""
    docs = web_retriever.search(topic, k=5)
    topic_lower = (topic or "").lower()
    is_service = any(kw in topic_lower for kw in _POLICY_SERVICE_TOPICS)
    suppress = (
        "\nFOOTER_HINT: SUPPRESS_LEAD_FOOTER (post-purchase policy question)."
        if is_service else ""
    )
    if not docs:
        return "NO_RESULTS: No policy information found for this topic." + suppress
    lines = [f"Policy info for '{topic}':"]
    for i, doc in enumerate(docs, 1):
        lines.append(f"\n[{i}] {doc.page_content[:800]}")
    if suppress:
        lines.append(suppress)
    return "\n".join(lines)


def tool_lookup_order_status(
    *,
    fetch_fn,
    build_response_fn,
    target_domain: str,
    order_id: str = "",
    email: str = "",
) -> str:
    """Look up live shipping status. fetch_fn = fetch_shopify_order_status."""
    if not order_id and not email:
        return "MISSING_INPUT: Need order number or email."
    tracking_data = fetch_fn(order_id, email, target_domain)
    return build_response_fn(tracking_data, target_domain)


def tool_capture_sales_lead(
    *,
    send_email_fn,
    customer_email: str,
    interest_summary: str,
    target_domain: str,
) -> str:
    """Forward a sales lead to the brand's sales inbox in the background."""
    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", customer_email or ""):
        return "INVALID_EMAIL: Could not parse a valid email address."
    threading.Thread(
        target=send_email_fn,
        args=(customer_email, interest_summary, "", target_domain),
        daemon=True,
    ).start()
    logger.info(f"📧 [Tool] Sales lead captured: {customer_email} on {target_domain}")
    return f"SUCCESS: Forwarded {customer_email} to sales team. They will respond within 24 hours."


def tool_get_showroom_info(*, target_domain: str) -> str:
    """Return the physical headquarters / showroom address.

    Customers asking "where is your showroom", "do you have a store", "company
    location", "address" etc. should be answered with this single source of truth.
    """
    try:
        # Imported here so the tools module stays decoupled from config side-effects.
        from config import COMPANY_ADDRESS
    except Exception:
        COMPANY_ADDRESS = "1001 W Crosby Rd, Carrollton, TX 75006"
    return (
        "Headquarters / Showroom:\n"
        f"Address: {COMPANY_ADDRESS}\n"
        "We recommend calling ahead to confirm availability before your visit. "
        "Use the SHOWROOM_ADDRESS value above verbatim in your reply."
    )


def _format_warranty_node(ticket_id: str, node: dict, is_start: bool = False) -> str:
    """Format a non-terminal warranty node for the LLM to paraphrase."""
    header = "WARRANTY_TICKET_STARTED" if is_start else "WARRANTY_CONTINUE"
    lines = [
        header,
        f"TICKET_ID: {ticket_id}",
        f"CURRENT_NODE: {node['node_id']}",
        f"NODE_TYPE: {node.get('type', '?')}",
        f"PROMPT: {node['prompt']}",
    ]
    options = node.get("options", [])
    if options:
        lines.append("OPTIONS (present these to customer):")
        for opt in options:
            lines.append(f"  - answer_key={opt.get('answer_key', '?')} | Label: {opt['label']}")
    else:
        lines.append("OPTIONS: (free text — accept any input)")
    lines += [
        "",
        "INSTRUCTION: Present the PROMPT to the customer in a warm, friendly tone.",
        "When the customer responds, map their answer to the closest answer_key and call answer_warranty_question.",
        "DO NOT make any warranty decision yourself. SUPPRESS_LEAD_FOOTER",
    ]
    return "\n".join(lines)


def _format_warranty_result(ticket_id: str, result: dict) -> str:
    """Format a warranty transition result for the LLM to paraphrase."""
    next_node = result["next_node"]
    is_terminal = result["is_terminal"]

    if is_terminal:
        action = next_node.get("action", "awaiting_admin")
        terminal_class = result.get("terminal_class", "awaiting_admin_review")
        evidence = result.get("evidence_required", [])
        evidence_email = result.get("evidence_email", "")
        internal_note = next_node.get("internal_note", "")
        lines = [
            "WARRANTY_TERMINAL_REACHED",
            f"TICKET_ID: {ticket_id}",
            f"ACTION: {action}",
            f"TERMINAL_CLASS: {terminal_class}",
            f"PROMPT_FOR_CUSTOMER: {next_node['prompt']}",
        ]
        if evidence:
            lines.append(f"EVIDENCE_REQUIRED: {', '.join(evidence)}")
        if evidence_email:
            lines.append(f"EVIDENCE_SEND_TO: {evidence_email}")
        lines.append(f"INTERNAL_NOTE: {internal_note}")
        lines += [""]
        if terminal_class == "awaiting_admin_review":
            evidence_hint = ""
            if "video_of_issue" in evidence:
                email = evidence_email or "service@osakititan.com"
                evidence_hint = (
                    f" Also ask the customer to send a photo or video of the issue to {email}."
                )
            lines.append(
                "INSTRUCTION: Deliver the PROMPT_FOR_CUSTOMER verbatim. "
                "DO NOT promise replacement, tech dispatch, compensation, refund, or any approval. "
                "The prompt already says 'our team will review' — keep that language exactly."
                + evidence_hint
                + " AWAITING_ADMIN_REVIEW=TRUE"
            )
        elif terminal_class == "send_info":
            lines.append(
                "INSTRUCTION: Deliver the PROMPT_FOR_CUSTOMER. "
                "This is a self-service step — you may briefly explain the how-to if helpful."
            )
        elif terminal_class in ("awaiting_evidence", "request_evidence"):
            lines.append(
                f"INSTRUCTION: Ask customer to send the evidence to {evidence_email or 'service@osakititan.com'}. "
                "Tell them someone will follow up after review."
            )
        else:
            lines.append("INSTRUCTION: Deliver the PROMPT_FOR_CUSTOMER verbatim.")
        lines.append("SUPPRESS_LEAD_FOOTER")
        return "\n".join(lines)

    # Non-terminal: next question
    lines = [
        "WARRANTY_CONTINUE",
        f"TICKET_ID: {ticket_id}",
        f"CURRENT_NODE: {next_node['node_id']}",
        f"NODE_TYPE: {next_node.get('type', '?')}",
        f"PROMPT: {next_node['prompt']}",
    ]
    options = next_node.get("options", [])
    if options:
        lines.append("OPTIONS (present these to customer):")
        for opt in options:
            lines.append(f"  - answer_key={opt.get('answer_key', '?')} | Label: {opt['label']}")
    else:
        lines.append("OPTIONS: (free text — accept any input)")
    lines += [
        "",
        "INSTRUCTION: Present PROMPT and OPTIONS to customer in a warm, friendly tone.",
        "When they respond, call warranty_answer with the matching answer_key.",
        "DO NOT make any warranty decision yourself. SUPPRESS_LEAD_FOOTER",
    ]
    return "\n".join(lines)


def tool_start_warranty_workflow(*, session_id: str, domain: str) -> str:
    """Start a new warranty intake session and return the first question node."""
    # Runtime validation
    if not session_id or not isinstance(session_id, str):
        return "WARRANTY_INPUT_ERROR: session_id must be a non-empty string."

    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(__file__))
        from warranty_workflow import WarrantyEngine
    except Exception as e:
        return f"WARRANTY_UNAVAILABLE: {e}"

    try:
        ticket_id, node = WarrantyEngine.start_session(session_id, domain or "unknown")
        logger.info(f"🎫 Warranty started — session={session_id} ticket={ticket_id} domain={domain}")
        return _format_warranty_node(ticket_id, node, is_start=True)
    except Exception as e:
        logger.error(f"start_warranty_workflow error: {e}")
        return f"WARRANTY_ERROR: {e}"


# Backward-compat alias (main.py references 'warranty_start' internally)
tool_warranty_start = tool_start_warranty_workflow


def tool_answer_warranty_question(*, ticket_id: str, answer_key: str) -> str:
    """Submit the customer's answer to the current warranty node and advance the workflow."""
    # Runtime validation
    if not ticket_id or not isinstance(ticket_id, str):
        return "WARRANTY_INPUT_ERROR: ticket_id must be a non-empty string."
    if not answer_key or not isinstance(answer_key, str):
        return "WARRANTY_INPUT_ERROR: answer_key must be a non-empty string."

    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(__file__))
        from warranty_workflow import WarrantyEngine
    except Exception as e:
        return f"WARRANTY_UNAVAILABLE: {e}"

    try:
        result = WarrantyEngine.submit_answer(ticket_id, answer_key)
        logger.info(
            f"🎫 Warranty answer — ticket={ticket_id} answer_key={answer_key} "
            f"next_node={result.get('next_node_id')} terminal={result.get('is_terminal')} "
            f"terminal_class={result.get('terminal_class')} "
            f"awaiting_admin={result.get('terminal_class') == 'awaiting_admin_review'}"
        )
        return _format_warranty_result(ticket_id, result)
    except ValueError as e:
        # Answer didn't match — give the LLM the valid options so it can ask again
        try:
            current_node = WarrantyEngine.get_current_node(ticket_id)
        except Exception:
            current_node = None
        options_hint = ""
        if current_node and current_node.get("options"):
            opts = [f"answer_key={o.get('answer_key')} | {o['label']}"
                    for o in current_node["options"]]
            options_hint = "\nVALID OPTIONS:\n  - " + "\n  - ".join(opts)
        return (
            f"WARRANTY_ANSWER_MISMATCH: {e}{options_hint}\n"
            f"INSTRUCTION: Ask the customer to clarify which option they meant and retry."
        )
    except Exception as e:
        logger.error(f"answer_warranty_question error: {e}")
        return f"WARRANTY_ERROR: {e}"


# Backward-compat alias
tool_warranty_answer = tool_answer_warranty_question


def tool_attach_warranty_evidence(
    *,
    ticket_id: str,
    evidence_type: str,
    original_filename: str = "",
    mime_type: str = "",
    file_size_bytes: int = 0,
) -> str:
    """
    Record evidence metadata for a warranty ticket.

    In this phase, no actual file is uploaded here — the binary upload
    is handled by POST /api/v1/warranty/{ticket_id}/evidence.
    This tool allows the LLM to acknowledge the evidence requirement
    and record what the customer has described submitting.
    """
    _ALLOWED_EVIDENCE_TYPES = {
        "damage_photos", "video_of_issue", "proof_of_purchase",
        "photo_of_chair", "photo_of_defect", "proof_of_delivery",
        "assembly_photo", "remote_photo", "other",
    }
    # Runtime validation
    if not ticket_id or not isinstance(ticket_id, str):
        return "WARRANTY_INPUT_ERROR: ticket_id must be a non-empty string."
    if not evidence_type or not isinstance(evidence_type, str):
        return "WARRANTY_INPUT_ERROR: evidence_type must be a non-empty string."
    if evidence_type not in _ALLOWED_EVIDENCE_TYPES:
        return (
            f"WARRANTY_INPUT_ERROR: Unknown evidence_type {evidence_type!r}. "
            f"Allowed: {sorted(_ALLOWED_EVIDENCE_TYPES)}"
        )

    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(__file__))
        from warranty_workflow import WarrantyEngine
    except Exception as e:
        return f"WARRANTY_UNAVAILABLE: {e}"

    try:
        ev = WarrantyEngine.record_evidence(
            ticket_id=ticket_id,
            evidence_type=evidence_type,
            file_path="",           # not uploaded yet
            original_filename=original_filename,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
        )
        logger.info(
            f"📎 Evidence noted — ticket={ticket_id} type={evidence_type} "
            f"file={original_filename or '(not uploaded yet)'}"
        )
        return (
            f"EVIDENCE_NOTED\n"
            f"TICKET_ID: {ticket_id}\n"
            f"EVIDENCE_TYPE: {evidence_type}\n"
            f"FILENAME: {original_filename or '(to be uploaded)'}\n"
            f"RECORD_ID: {ev.id}\n"
            f"INSTRUCTION: Let the customer know their evidence has been noted. "
            f"Ask them to upload the file via the evidence upload link or email it to service@osakititan.com."
        )
    except ValueError as e:
        return f"WARRANTY_ERROR: {e}"
    except Exception as e:
        logger.error(f"attach_warranty_evidence error: {e}")
        return f"WARRANTY_ERROR: {e}"


def tool_escalate_to_human(
    *,
    contact_msg_fn,
    target_domain: str,
    reason: str = "general",
) -> str:
    """Hand off to a human rep with the appropriate phone for the brand.

    Sales-related → brand sales line. Everything else (warranty, repair,
    cancellation, general) → service/support line. Service answers are
    flagged with SUPPRESS_LEAD_FOOTER so we don't pitch sales after them.
    """
    sales_reasons = {"sales", "pricing", "discount"}
    routing = "PRODUCTS" if reason in sales_reasons else "QA"
    msg = contact_msg_fn(routing, target_domain)
    if reason not in sales_reasons:
        msg += "\nFOOTER_HINT: SUPPRESS_LEAD_FOOTER (handed off to service)."
    return msg


# ============================================================================
# Tool schemas (passed to OpenAI as `tools=[...]`)
# ============================================================================

TOOL_SCHEMAS: List[Any] = [
    {
        "type": "function",
        "function": {
            "name": "search_chair_specs",
            "description": (
                "Look up factual spec details (dimensions, door size, weight, features, "
                "price, warranty bundled with the model) for a specific massage chair. "
                "ALWAYS call this when the user asks about a particular model — even if "
                "they only give part of the name like 'Hypnos' or 'Nova II'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The chair model the user mentioned, or '' if unclear.",
                    },
                    "spec_topic": {
                        "type": "string",
                        "description": "The specific spec the user is asking about, e.g. 'minimum door size', 'dimensions', 'features', 'price'.",
                    },
                },
                "required": ["model_name", "spec_topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_chairs",
            "description": (
                "Recommend premium full-body massage chairs based on the user's needs. "
                "Use when the user asks 'recommend', 'best chair', 'which one should I buy', "
                "or wants to compare options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_need": {
                        "type": "string",
                        "description": "What the customer is looking for, e.g. 'tall person 6'5\"', 'back pain relief', 'compact for small space'.",
                    },
                    "budget_min": {"type": "number", "description": "Minimum budget in USD. Omit if not specified."},
                    "budget_max": {"type": "number", "description": "Maximum budget in USD. Omit if not specified."},
                    "exclude_models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Models the user has already seen / rejected. Use this to vary recommendations on follow-up turns.",
                    },
                    "num_recommendations": {"type": "integer", "default": 3},
                },
                "required": ["user_need"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order_status",
            "description": (
                "Get real-time shipping status, hub, and ETA for a customer's order. "
                "Requires order number and/or the email used at checkout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order number, e.g. 'OSKUS11308' or '#1234'. Empty string if unknown."},
                    "email": {"type": "string", "description": "Customer email used at checkout. Empty string if unknown."},
                },
                "required": ["order_id", "email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repair_help",
            "description": (
                "Look up troubleshooting steps for an error code, error symptom, "
                "assembly question, or general repair guidance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_description": {"type": "string"},
                    "error_code": {"type": "string", "description": "If user mentions a code like '63', '63.0', 'E5', use it. Empty otherwise."},
                },
                "required": ["issue_description", "error_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_warranty_or_policy",
            "description": (
                "Look up policy text: warranty terms, return policy, shipping/delivery, "
                "white-glove service, installation, sales/discount policy, FAQ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Specific topic, e.g. 'warranty length', 'return policy', 'white glove delivery'.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_sales_lead",
            "description": (
                "Capture a customer's email so the sales team can follow up within 24 hours. "
                "Call ONLY after the user has volunteered their email address in response to a sales offer. "
                "Do NOT use for shipping-tracking emails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {"type": "string"},
                    "interest_summary": {
                        "type": "string",
                        "description": "Brief summary of what the customer was interested in, taken from earlier turns.",
                    },
                },
                "required": ["customer_email", "interest_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Provide the correct phone number / contact info for human help. "
                "Use when the customer asks for an agent, can't resolve via chat, or for complex situations like cancellations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": ["sales", "pricing", "discount", "warranty", "repair", "cancellation", "general"],
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_warranty_workflow",
            "description": (
                "Start a structured warranty intake process when a customer reports "
                "a product defect, delivery damage, installation difficulty, or any "
                "issue that may be covered under warranty. "
                "Call this instead of get_repair_help when the customer is reporting "
                "a problem that may need a replacement part, technician, or compensation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_hint": {
                        "type": "string",
                        "description": "Brief one-sentence description of what the customer reported.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_warranty_question",
            "description": (
                "Submit the customer's answer to the current warranty question and advance "
                "the workflow to the next step. Use the answer_key that best matches what "
                "the customer said. The tool will return the next question or a terminal action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The WARRANTY_TICKET_ID from the active warranty session.",
                    },
                    "answer_key": {
                        "type": "string",
                        "description": (
                            "The answer_key of the option that matches the customer's response. "
                            "Must exactly match one of the answer_key values listed in OPTIONS."
                        ),
                    },
                },
                "required": ["ticket_id", "answer_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attach_warranty_evidence",
            "description": (
                "Note that the customer has described or is about to submit evidence "
                "(photo, video, receipt) for their warranty ticket. "
                "Call this when a terminal node requires evidence and the customer "
                "indicates they have the file ready. "
                "This records metadata only — the actual file upload is done separately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The WARRANTY_TICKET_ID from the active warranty session.",
                    },
                    "evidence_type": {
                        "type": "string",
                        "enum": [
                            "damage_photos", "video_of_issue", "proof_of_purchase",
                            "photo_of_chair", "photo_of_defect", "proof_of_delivery",
                            "assembly_photo", "remote_photo", "other",
                        ],
                        "description": "Category of evidence the customer is submitting.",
                    },
                    "original_filename": {
                        "type": "string",
                        "description": "Filename the customer mentioned, if any. Empty if not stated.",
                    },
                },
                "required": ["ticket_id", "evidence_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_showroom_info",
            "description": (
                "Return the official Carrollton, TX headquarters / showroom address. "
                "Use this whenever the customer asks where the showroom / store / office / "
                "company is located, or wants to visit / see chairs in person."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Warranty-mode tool subset
# When an active warranty ticket is in progress, the agent must only use
# these tools so it cannot veer into free-form decisions.
# ---------------------------------------------------------------------------
WARRANTY_TOOL_SCHEMAS: List[Any] = [
    s for s in TOOL_SCHEMAS
    if s["function"]["name"] in {
        "answer_warranty_question",
        "attach_warranty_evidence",
        "escalate_to_human",
        "get_warranty_or_policy",
    }
]

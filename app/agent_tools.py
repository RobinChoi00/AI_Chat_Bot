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
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

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

        # Dense search
        try:
            dense_hits = self.vectorstore.similarity_search(query, k=k * 2)
        except Exception as e:
            logger.warning(f"dense search failed: {e}")
            dense_hits = []

        # BM25 search
        bm25_hits: List[Document] = []
        if self.bm25 is not None:
            try:
                tokens = self._tokenize(query)
                scores = self.bm25.get_scores(tokens)
                # top k*2 by score
                ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: k * 2]
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


def tool_search_chair_specs(
    *,
    products_retriever: HybridRetriever,
    query: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Look up spec details for one or more chairs. Use when the user asks about
    dimensions, weight, door size, features, price etc. of a specific model.
    """
    search_query = f"{model_name} {query}".strip() if model_name else query
    docs = products_retriever.search(search_query, k=5, model_hint=model_name)
    if not docs:
        return "NO_RESULTS: No matching specifications found in the catalog."

    lines: List[str] = []

    for i, doc in enumerate(docs, 1):
        title = _extract_title(doc)

        # Extract AUTHORITATIVE lines that directly answer the query
        auth_lines = _find_authoritative_lines(doc.page_content, query)
        if auth_lines:
            lines.append(f"\n--- Result {i}: {title} ---")
            lines.append("AUTHORITATIVE SPEC VALUES (use EXACTLY these numbers, do not paraphrase):")
            for al in auth_lines:
                lines.append(f"  {al}")
            # Also include full body (truncated) for context
            lines.append("\nFull spec context:")
            lines.append(doc.page_content[:1200])
        else:
            lines.append(f"\n--- Result {i}: {title} ---")
            lines.append(doc.page_content[:1500])

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

TOOL_SCHEMAS: List[Dict[str, Any]] = [
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
]

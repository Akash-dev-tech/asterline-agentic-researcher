"""Pre-execution input scrubbing and deterministic safety guardrails.

Intercepts raw user input before it reaches any database, LLM prompt,
or LangSmith trace boundary to enforce zero-trust data sanitization.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ----------------------------------------------------------------------
# 1. Secret & Token Detection Rules
# ----------------------------------------------------------------------
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # HTTP Authorization header (Bearer or Basic or raw token)
    (
        re.compile(r"(?i)(authorization\s*:\s*(?:bearer\s+)?)([a-zA-Z0-9_\-\.]{8,})"),
        r"\1[REDACTED_SECRET]",
    ),
    # Standalone Bearer token
    (re.compile(r"(?i)\b(bearer\s+)([a-zA-Z0-9_\-\.]{12,})\b"), r"\1[REDACTED_SECRET]"),
    # Prefixed API keys: ast_..., sk-..., gsk_..., pk_..., key_..., token_..., secret_...
    (
        re.compile(
            r"\b(?:ast|gsk|sk|pk|key|token|secret)[_-][a-zA-Z0-9_\-]{6,}\b",
            re.IGNORECASE,
        ),
        "[REDACTED_SECRET]",
    ),
    # Explicit ingestion key assignments (e.g. "ingestion key is abc...", "ingestion_key: abc...")
    (
        re.compile(r"(?i)(ingestion[\s_-]*key\s*(?:is|=|:)\s*)([a-zA-Z0-9_\-]{8,})"),
        r"\1[REDACTED_SECRET]",
    ),
    # Generic credential assignments (e.g. "api_key = abc...", "password: secret")
    (
        re.compile(
            r"(?i)((?:api[\s_-]*key|client[\s_-]*secret|access[\s_-]*token|password)\s*(?:is|=|:)\s*)([^\s,;]{6,})"
        ),
        r"\1[REDACTED_SECRET]",
    ),
]

# ----------------------------------------------------------------------
# 2. PII Redaction Rules
# ----------------------------------------------------------------------
# Introductory patterns identifying user personal names
PII_INTRO_PATTERN = re.compile(
    r"\b(?i:(?P<intro>i\s+am|i\'m|my\s+name\s+is|this\s+is))\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

# Common words following "I am" that are NOT personal names
PII_EXCLUSIONS = frozenset(
    {
        "a",
        "an",
        "the",
        "on",
        "in",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "using",
        "trying",
        "working",
        "looking",
        "wondering",
        "having",
        "experiencing",
        "facing",
        "testing",
        "running",
        "building",
        "getting",
        "writing",
        "asking",
        "currently",
        "presently",
        "also",
        "just",
        "already",
        "now",
        "still",
        "ready",
        "new",
        "interested",
        "aware",
        "sure",
        "unsure",
        "responsible",
        "launch",
        "team",
        "enterprise",
        "free",
        "pro",
        "admin",
        "user",
        "member",
    }
)

# ----------------------------------------------------------------------
# 3. Medical Context Flagging Rules
# ----------------------------------------------------------------------
MEDICAL_KEYWORDS = (
    "surgery",
    "diagnosis",
    "medical",
    "hospital",
    "prescription",
    "doctor note",
    "doctor's note",
    "therapy",
    "medication",
    "physician",
    "clinic",
    "chemotherapy",
    "treatment",
    "illness",
    "disease",
    "patient",
    "health condition",
)

# ----------------------------------------------------------------------
# 4. Company-Fact Override Protection Rules
# ----------------------------------------------------------------------
COMPANY_OVERRIDE_PATTERNS = [
    # Commands attempting to store/record company facts into user memory
    re.compile(
        r"(?i)\b(?:remember|store|save|record|note|set|update)\s+(?:that\s+)?"
        r"(?:the\s+)?(?:launch|team|enterprise|company|asterline|pricing|plan|policy|retention|stipend|sla|limit)"
    ),
    # Explicit override assertions against documentation/knowledge base
    re.compile(
        r"(?i)\b(?:override|disregard|ignore)\s+(?:the\s+)?"
        r"(?:official|company|documentation|knowledge\s*base|handbook|runbook|policy|pricing)"
    ),
    # Assertions declaring false pricing or terms
    re.compile(
        r"(?i)\b(?:the\s+)?(?:launch|team|enterprise)\s+plan\s+(?:costs?|is|price\s+is)\s+[\$£€]?\d+"
    ),
    # Direct declarations of corporate policy overrides
    re.compile(r"(?i)\b(?:company|asterline)\s+policy\s+(?:is|states|now)\b"),
]


def scrub_user_input(text: str) -> Tuple[str, List[str]]:
    """Sanitize user input before it reaches databases, LLM prompts, or LangSmith traces.

    Applies deterministic regex rules to redact secrets and PII, and flags
    medical context or unauthorized company-fact override attempts.

    Args:
        text: Raw user input text.

    Returns:
        A tuple of (sanitized_text, warnings) where warnings is a list of
        triggered warning identifier strings.
    """
    scrubbed = text
    warnings: List[str] = []

    # 1. Secret & Token Redaction
    for pattern, replacement in SECRET_PATTERNS:
        if pattern.search(scrubbed):
            scrubbed = pattern.sub(replacement, scrubbed)
            if "secret_redacted" not in warnings:
                warnings.append("secret_redacted")

    # 2. PII Redaction
    def _pii_repl(match: re.Match[str]) -> str:
        intro = match.group("intro")
        name = match.group("name")
        words = name.split()
        if any(w.lower() in PII_EXCLUSIONS for w in words):
            first_w = words[0]
            if len(words) > 1 and first_w.lower() not in PII_EXCLUSIONS:
                rest = " ".join(words[1:])
                return f"{intro} [REDACTED_PII] {rest}"
            return match.group(0)
        return f"{intro} [REDACTED_PII]"

    if PII_INTRO_PATTERN.search(scrubbed):
        new_scrubbed = PII_INTRO_PATTERN.sub(_pii_repl, scrubbed)
        if new_scrubbed != scrubbed:
            scrubbed = new_scrubbed
            if "pii_redacted" not in warnings:
                warnings.append("pii_redacted")

    # 3. Medical Context Flagging
    lower_text = scrubbed.lower()
    if any(
        re.search(r"\b" + re.escape(word) + r"\b", lower_text)
        for word in MEDICAL_KEYWORDS
    ):
        if "contains_medical_context" not in warnings:
            warnings.append("contains_medical_context")

    # 4. Company-Fact Override Protection
    if any(p.search(scrubbed) for p in COMPANY_OVERRIDE_PATTERNS):
        if "company_fact_override_attempt" not in warnings:
            warnings.append("company_fact_override_attempt")

    return scrubbed, warnings

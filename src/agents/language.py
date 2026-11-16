"""Hybrid Language and Egyptian Arabic Dialect Detector.

Solves Technical Risk R3:
- langdetect is unreliable on short text and cannot differentiate Egyptian Arabic (ar_eg) from MSA (ar).
- Uses Unicode script ranges, colloquial Egyptian lexical markers, and Franco-Arabizi patterns.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import langdetect
    from langdetect import DetectorFactory
    # Enforce deterministic results
    DetectorFactory.seed = 0
except ImportError:
    langdetect = None

# Egyptian Arabic dialect lexical markers (keywords, particles, pronouns)
# Supports optional Arabic conjunction prefixes و (and) or ف (so)
EGYPTIAN_KEYWORDS = [
    r"عايز[ةه]?",
    r"عاوز[ةه]?",
    r"إ?زاي",
    r"فين",
    r"أ?د\s*إ?يه",
    r"قد\s*إ?يه",
    r"كده",
    r"عل?شان",
    r"النهارده?",
    r"بتاع(ي|نا|هم|ها|ك|كم)?",
    r"دلوقتي",
    r"مش",
    r"برض[وه]",
    r"أ?وي",
    r"كوتش",
    r"يا\s*بطل",
    r"ثواني",
    r"حاجة",
    r"وشي",
    r"ليه",
    r"كويس",
    r"خالص",
    r"معلش",
    r"ابعتلي",
    r"إ?متى",
    r"كام",
]

# Match either start of string, whitespace, punctuation, or conjunction prefix و/ف
EGYPTIAN_REGEX = re.compile(
    r"(?:^|[\s،.!?؟])(?:[وف])?(?:" + "|".join(EGYPTIAN_KEYWORDS) + r")(?=[\s،.!?؟]|$)",
    re.IGNORECASE,
)
ARABIC_SCRIPT_REGEX = re.compile(r"[\u0600-\u06FF]")
ARABIZI_NUM_REGEX = re.compile(r"\b[a-zA-Z]*[23578][a-zA-Z]*\b")


def detect_language(text: str) -> str:
    """Detects text language code: 'en', 'ar' (MSA), or 'ar_eg' (Egyptian Arabic).

    Args:
        text: Input customer message.

    Returns:
        One of 'en', 'ar', or 'ar_eg'. Defaults to 'en' if undetermined.
    """
    clean_text = text.strip()
    if not clean_text:
        return "en"

    # 1. Unicode Arabic Script Ratio Check
    arabic_chars = len(ARABIC_SCRIPT_REGEX.findall(clean_text))
    total_letters = sum(1 for c in clean_text if c.isalpha() or ARABIC_SCRIPT_REGEX.match(c))

    if total_letters > 0 and (arabic_chars / total_letters) > 0.3:
        # Text is in Arabic script; now evaluate dialect
        if EGYPTIAN_REGEX.search(clean_text):
            return "ar_eg"
        return "ar"

    # 2. Check for Arabizi (Franco-Arabic written in Latin with numbers 2, 3, 7, etc.)
    if ARABIZI_NUM_REGEX.search(clean_text):
        return "ar_eg"

    # 3. Fallback to langdetect for Latin and other scripts
    if langdetect:
        try:
            detected = langdetect.detect(clean_text)
            if detected.startswith("ar"):
                return "ar_eg" if EGYPTIAN_REGEX.search(clean_text) else "ar"
            if detected.startswith("en"):
                return "en"
        except Exception:
            pass

    # Default fallback
    return "en"

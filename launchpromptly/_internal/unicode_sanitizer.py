"""
Invisible Unicode & encoding attack detection module.
Detects and optionally strips malicious Unicode characters used in LLM attacks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set

MAX_SCAN_LENGTH = 1024 * 1024  # 1MB

UnicodeSanitizeAction = Literal["strip", "warn", "block"]
UnicodeThreatType = Literal["zero_width", "bidi_override", "tag_char", "homoglyph", "variation_selector"]


@dataclass
class UnicodeSanitizeOptions:
    """Options for unicode sanitization."""

    action: Optional[UnicodeSanitizeAction] = None  # default: 'strip'
    detect_homoglyphs: Optional[bool] = None  # default: True


@dataclass
class UnicodeThreat:
    type: UnicodeThreatType
    position: int
    char: str
    code_point: str  # e.g. "U+200B"


@dataclass
class UnicodeScanResult:
    found: bool
    threats: List[UnicodeThreat]
    sanitized_text: Optional[str] = None


# -- Cyrillic -> Latin homoglyph map -------------------------------------------

_CYRILLIC_TO_LATIN: Dict[str, str] = {
    "\u0430": "a",  # а -> a
    "\u0435": "e",  # е -> e
    "\u043E": "o",  # о -> o
    "\u0441": "c",  # с -> c
    "\u0440": "p",  # р -> p
    "\u0443": "y",  # у -> y
    "\u0445": "x",  # х -> x
    "\u0412": "B",  # В -> B
    "\u041D": "H",  # Н -> H
    "\u041A": "K",  # К -> K
    "\u041C": "M",  # М -> M
    "\u0422": "T",  # Т -> T
    "\u0410": "A",  # А -> A
    "\u0415": "E",  # Е -> E
    "\u041E": "O",  # О -> O
    "\u0421": "C",  # С -> C
    "\u0420": "P",  # Р -> P
}

# -- Regex patterns for threat categories --------------------------------------

_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF\u2060\u00AD]")
_BIDI_OVERRIDE_RE = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
_VARIATION_SELECTOR_RE = re.compile(r"[\uFE00-\uFE0F]")

_LATIN_RE = re.compile(r"[a-zA-Z]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


# -- Helpers -------------------------------------------------------------------

def _to_code_point(char: str) -> str:
    cp = ord(char)
    return "U+" + format(cp, "X").zfill(4)


# -- Detection -----------------------------------------------------------------

def scan_unicode(
    text: str,
    options: Optional[UnicodeSanitizeOptions] = None,
) -> UnicodeScanResult:
    """Scan text for malicious Unicode characters.
    Returns detected threats and optionally a sanitized version of the text.
    Text longer than 1MB is truncated before scanning.
    """
    action = (options.action if options and options.action is not None else "strip")
    detect_homoglyphs = (options.detect_homoglyphs if options and options.detect_homoglyphs is not None else True)

    if not text:
        return UnicodeScanResult(found=False, threats=[])

    # Cap input length to prevent DoS
    scan_text = text[:MAX_SCAN_LENGTH] if len(text) > MAX_SCAN_LENGTH else text

    threats: List[UnicodeThreat] = []

    # Single-character threat detection: iterate through every code point
    i = 0
    while i < len(scan_text):
        char = scan_text[i]
        cp = ord(char)

        # Handle tag characters (astral plane U+E0001 to U+E007F)
        # In Python, surrogate pairs are handled automatically — each code point
        # is a single character in the string.
        if 0xE0001 <= cp <= 0xE007F:
            threats.append(UnicodeThreat(
                type="tag_char",
                position=i,
                char=char,
                code_point=_to_code_point(char),
            ))
            i += 1
            continue

        if _ZERO_WIDTH_RE.match(char):
            threats.append(UnicodeThreat(
                type="zero_width",
                position=i,
                char=char,
                code_point=_to_code_point(char),
            ))
        elif _BIDI_OVERRIDE_RE.match(char):
            threats.append(UnicodeThreat(
                type="bidi_override",
                position=i,
                char=char,
                code_point=_to_code_point(char),
            ))
        elif _VARIATION_SELECTOR_RE.match(char):
            threats.append(UnicodeThreat(
                type="variation_selector",
                position=i,
                char=char,
                code_point=_to_code_point(char),
            ))

        i += 1

    # Homoglyph detection: check each word for mixed Cyrillic + Latin
    if detect_homoglyphs:
        words = scan_text.split()
        pos = 0
        for word in words:
            # Find actual position of this word in the scan text
            word_start = scan_text.find(word, pos)
            if word_start == -1:
                pos += len(word) + 1
                continue

            has_latin = bool(_LATIN_RE.search(word))
            has_cyrillic = bool(_CYRILLIC_RE.search(word))

            if has_latin and has_cyrillic:
                # Flag each Cyrillic char within this mixed-script word
                for j in range(len(word)):
                    ch = word[j]
                    if _CYRILLIC_RE.match(ch):
                        threats.append(UnicodeThreat(
                            type="homoglyph",
                            position=word_start + j,
                            char=ch,
                            code_point=_to_code_point(ch),
                        ))

            pos = word_start + len(word)

    # Sort threats by position for deterministic output
    threats.sort(key=lambda t: t.position)

    found = len(threats) > 0

    # Build sanitized text only for 'strip' action
    sanitized_text: Optional[str] = None
    if action == "strip" and found:
        # Build a set of positions to remove (non-homoglyph threats)
        remove_positions: Set[int] = set()
        homoglyph_replacements: Dict[int, str] = {}

        for threat in threats:
            if threat.type == "homoglyph":
                replacement = _CYRILLIC_TO_LATIN.get(threat.char)
                if replacement:
                    homoglyph_replacements[threat.position] = replacement
            else:
                remove_positions.add(threat.position)

        chars: List[str] = []
        for idx in range(len(scan_text)):
            if idx in remove_positions:
                continue
            replacement = homoglyph_replacements.get(idx)
            if replacement is not None:
                chars.append(replacement)
            else:
                chars.append(scan_text[idx])
        sanitized_text = "".join(chars)

    return UnicodeScanResult(found=found, threats=threats, sanitized_text=sanitized_text)

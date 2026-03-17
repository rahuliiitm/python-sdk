"""Locale-specific content filter patterns."""

from __future__ import annotations

import re
from typing import Dict, List

from . import LocaleContentPattern

ES_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"\b(?:malditos?|asquerosos?)\s+(?:negros?|indios?|musulmanes?|judíos?)\b", re.I),
            re.compile(r"\b(?:muerte\s+a\s+(?:los|las)\s+\w+)\b", re.I),
            re.compile(r"\b(?:raza\s+(?:inferior|superior))\b", re.I),
            re.compile(r"\b(?:limpieza\s+étnica)\b", re.I),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"\b(?:cómo\s+(?:hacer|fabricar|construir)\s+(?:una?\s+)?(?:bomba|explosivo|arma))\b", re.I),
            re.compile(r"\b(?:instrucciones\s+para\s+(?:matar|asesinar|envenenar))\b", re.I),
        ],
        severity="block",
    ),
]

PT_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"\b(?:morte\s+a(?:os?)?\s+\w+)\b", re.I),
            re.compile(r"\b(?:raça\s+(?:inferior|superior))\b", re.I),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"\b(?:como\s+(?:fazer|fabricar)\s+(?:uma?\s+)?(?:bomba|explosivo|arma))\b", re.I),
        ],
        severity="block",
    ),
]

ZH_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"(?:杀光|灭绝|消灭)\s*(?:所有|全部)"),
            re.compile(r"(?:种族清洗|种族灭绝)"),
            re.compile(r"(?:劣等|低等)\s*(?:民族|种族|人种)"),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"(?:如何|怎么|怎样)\s*(?:制造|制作)\s*(?:炸弹|爆炸物|武器)"),
            re.compile(r"(?:毒杀|暗杀|谋杀)\s*(?:方法|教程|指南)"),
        ],
        severity="block",
    ),
]

JA_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"(?:殺せ|死ね|消えろ)\s*(?:全員|みんな)"),
            re.compile(r"(?:民族浄化|人種差別)"),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"(?:爆弾|爆発物|武器)\s*(?:の作り方|を作る方法|の製造)"),
        ],
        severity="block",
    ),
]

KO_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"(?:죽여|없애)\s*(?:버려|라)"),
            re.compile(r"(?:인종\s*청소|민족\s*말살)"),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"(?:폭탄|폭발물|무기)\s*(?:만드는\s*법|제조\s*방법)"),
        ],
        severity="block",
    ),
]

DE_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"\b(?:Tod\s+(?:den|allen)\s+\w+)\b", re.I),
            re.compile(r"\b(?:(?:ethnische|rassische)\s+Säuberung)\b", re.I),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"\b(?:(?:Anleitung|Anweisungen)\s+(?:zum|zur)\s+(?:Töten|Morden|Vergiften))\b", re.I),
            re.compile(r"\b(?:(?:Bombe|Sprengstoff|Waffe)\s+(?:bauen|herstellen|basteln))\b", re.I),
        ],
        severity="block",
    ),
]

FR_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"\b(?:mort\s+aux?\s+\w+)\b", re.I),
            re.compile(r"\b(?:nettoyage\s+ethnique)\b", re.I),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"\b(?:comment\s+(?:fabriquer|construire)\s+(?:une?\s+)?(?:bombe|explosif|arme))\b", re.I),
        ],
        severity="block",
    ),
]

AR_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"(?:الموت\s+ل|اقتلوا\s+كل)"),
            re.compile(r"(?:تطهير\s+عرقي|إبادة\s+جماعية)"),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"(?:كيفية?\s+(?:صنع|تصنيع)\s+(?:قنبلة|متفجرات|سلاح))"),
        ],
        severity="block",
    ),
]

HI_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"(?:मार\s+डालो|खत्म\s+करो)\s+(?:सब|सभी)"),
            re.compile(r"(?:जातीय\s+सफाई|नरसंहार)"),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"(?:बम|विस्फोटक|हथियार)\s+(?:बनाने\s+का\s+तरीका|कैसे\s+बनाएं)"),
        ],
        severity="block",
    ),
]

RU_PATTERNS: List[LocaleContentPattern] = [
    LocaleContentPattern(
        category="hate_speech",
        patterns=[
            re.compile(r"(?:смерть\s+(?:всем|им))", re.I),
            re.compile(r"(?:этническ(?:ая|ие)\s+чистк(?:а|и))", re.I),
            re.compile(r"(?:(?:низшая|высшая)\s+раса)", re.I),
        ],
        severity="block",
    ),
    LocaleContentPattern(
        category="violence",
        patterns=[
            re.compile(r"(?:как\s+(?:сделать|изготовить)\s+(?:бомбу|взрывчатку|оружие))", re.I),
        ],
        severity="block",
    ),
]

LOCALE_PATTERNS: Dict[str, List[LocaleContentPattern]] = {
    "es": ES_PATTERNS,
    "pt": PT_PATTERNS,
    "zh": ZH_PATTERNS,
    "ja": JA_PATTERNS,
    "ko": KO_PATTERNS,
    "de": DE_PATTERNS,
    "fr": FR_PATTERNS,
    "ar": AR_PATTERNS,
    "hi": HI_PATTERNS,
    "ru": RU_PATTERNS,
}

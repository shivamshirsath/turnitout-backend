import re
import math
from typing import Dict, Optional

CONTRACTIONS_MAP = {
    r"\bI am\b": "I'm",
    r"\bI have\b": "I've",
    r"\bI will\b": "I'll",
    r"\bI would\b": "I'd",
    r"\bit is\b": "it's",
    r"\bit has\b": "it's",
    r"\bit will\b": "it'll",
    r"\bthat is\b": "that's",
    r"\bthere is\b": "there's",
    r"\bwhat is\b": "what's",
    r"\bwho is\b": "who's",
    r"\bdo not\b": "don't",
    r"\bdoes not\b": "doesn't",
    r"\bdid not\b": "didn't",
    r"\bcannot\b": "can't",
    r"\bcan not\b": "can't",
    r"\bwill not\b": "won't",
    r"\bwould not\b": "wouldn't",
    r"\bshould not\b": "shouldn't",
    r"\bcould not\b": "couldn't",
    r"\bwe are\b": "we're",
    r"\bthey are\b": "they're",
    r"\byou are\b": "you're",
    r"\bwe have\b": "we've",
    r"\bthey have\b": "they've",
    r"\byou have\b": "you've"
}

EXPANSIONS_MAP = {v: k.replace(r"\b", "") for k, v in CONTRACTIONS_MAP.items()}


class EmpiricalVoiceMatcher:
    """
    Universal Voice & Contraction Conditioner.
    Ensures first-person ("I'm", "I've") and third-person contractions match natural
    human writing across single-line and multi-paragraph inputs.
    """

    @staticmethod
    def extract_profile(sample_text: str) -> Dict[str, float]:
        if not sample_text or not sample_text.strip():
            return {
                "avg_sentence_len": 15.0,
                "std_sentence_len": 6.5,
                "contraction_rate": 0.0,
                "active": False
            }

        sentences = [s.strip() for s in re.split(r'[.!?]+', sample_text) if s.strip()]
        words = sample_text.split()
        if not sentences or not words:
            return {
                "avg_sentence_len": 15.0,
                "std_sentence_len": 6.5,
                "contraction_rate": 0.0,
                "active": False
            }

        lengths = [len(s.split()) for s in sentences]
        mean_l = sum(lengths) / len(lengths)
        var_l = sum((l - mean_l) ** 2 for l in lengths) / max(len(lengths), 1)
        std_l = math.sqrt(var_l)

        contractions = re.findall(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", sample_text, re.IGNORECASE)
        c_rate = (len(contractions) / len(words)) * 100.0

        return {
            "avg_sentence_len": round(mean_l, 1),
            "std_sentence_len": round(std_l, 1),
            "contraction_rate": round(c_rate, 2),
            "active": True
        }

    @staticmethod
    def apply_voice(
        text: str,
        profile: Optional[Dict[str, float]] = None,
        register: str = "everyday"
    ) -> str:
        reg = register.lower()
        use_contractions = False
        expand_contractions = False

        if profile and profile.get("active"):
            if profile["contraction_rate"] >= 1.0:
                use_contractions = True
            elif profile["contraction_rate"] < 0.2:
                expand_contractions = True
        else:
            if reg in ("everyday", "conversational", "executive"):
                use_contractions = True
            elif reg == "academic":
                expand_contractions = True

        if use_contractions:
            for pattern, repl in CONTRACTIONS_MAP.items():
                def _sub(m):
                    orig = m.group(0)
                    if repl.startswith("I'"):
                        return repl
                    return repl.capitalize() if orig[0].isupper() else repl
                text = re.sub(pattern, _sub, text, flags=re.IGNORECASE)
        elif expand_contractions:
            for contr, exp in EXPANSIONS_MAP.items():
                pattern = rf"\b{re.escape(contr)}\b"
                def _sub_exp(m):
                    orig = m.group(0)
                    if exp.startswith("I "):
                        return exp
                    return exp.capitalize() if orig[0].isupper() else exp
                text = re.sub(pattern, _sub_exp, text, flags=re.IGNORECASE)

        return text


def analyze_voice_profile(sample_text: str) -> Dict[str, float]:
    return EmpiricalVoiceMatcher.extract_profile(sample_text)
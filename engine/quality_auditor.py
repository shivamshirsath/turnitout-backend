import re
from typing import List, Set, Tuple
import nltk

NEGATION_TOKENS = {
    "no", "not", "never", "neither", "nor", "none", "nothing", "nowhere",
    "nobody", "without", "cannot", "can't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't",
    "wasn't", "weren't", "haven't", "hasn't", "hadn't"
}

BROKEN_GRAMMAR_PATTERNS = [
    r"\b(?:a|an|the)\s+(?:is|are|was|were|can|will|must|should)\b",
    r"\b(?:is|are|was|were)\s+(?:is|are|was|were)\b",
    r"\bto\s+(?:is|are|was|were|has|had|can|will|must)\b",
    r"\b(?:can|could|may|might|must|should|will|would|do|does|did)\s+[a-z]{3,}ed\b",
    r"\b(?:by|without|for|of|in|on|at|from|with)\s*[.,;!?]",
    r"\b(?:and|or|but)\s*[.;!?]"
]


class PropositionQualityAuditor:
    """
    v22.0 Precision Proposition & Grammar Auditor.
    Allows deep syntactic and lexical rewriting across every clause while strictly
    protecting numbers, acronyms, invariant tokens, and negation polarity.
    """

    def __init__(self):
        self.compiled_broken = [re.compile(p, re.IGNORECASE) for p in BROKEN_GRAMMAR_PATTERNS]

    @staticmethod
    def _extract_words(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9'-]+\b", text)

    @staticmethod
    def _extract_numbers_and_symbols(text: str) -> List[str]:
        return re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", text)

    @staticmethod
    def _extract_true_acronyms_and_invariants(text: str) -> Set[str]:
        tokens = text.split()
        entities = set()
        for tok in tokens:
            clean = tok.strip(".,;:!?()[]\"'")
            if not clean:
                continue
            if clean.startswith("ZXQINV"):
                entities.add(clean)
            elif len(clean) >= 2 and sum(1 for c in clean if c.isupper()) >= 2:
                entities.add(clean)
        return entities

    @staticmethod
    def _count_negations(words_lower: List[str]) -> int:
        return sum(1 for w in words_lower if w in NEGATION_TOKENS or w.endswith("n't"))

    @staticmethod
    def fix_indefinite_articles(text: str) -> str:
        """Automatically repairs 'a' vs 'an' after synonym replacement instead of rolling back!"""
        def _fix_a(m):
            art, space, nxt = m.group(1), m.group(2), m.group(3)
            low_n = nxt.lower()
            vowel_sound = (
                low_n[0] in "aeiou"
                and not low_n.startswith(("uni", "use", "usa", "usu", "one", "once", "eu"))
            ) or low_n.startswith(("hour", "honest", "honor"))
            target = "an" if vowel_sound else "a"
            if art[0].isupper():
                target = target.capitalize()
            return f"{target}{space}{nxt}"

        return re.sub(r"\b(a|an)(\s+)([A-Za-z]{2,})\b", _fix_a, text, flags=re.IGNORECASE)

    def verify_sentence_completeness_and_grammar(self, sentence: str) -> bool:
        s = sentence.strip()
        if not s or len(s.split()) < 3:
            return False
        if s[-1] not in ".!?":
            return False
        if s.count("(") != s.count(")") or s.count("[") != s.count("]"):
            return False
        if s.count('"') % 2 != 0:
            return False

        for pat in self.compiled_broken:
            if pat.search(s):
                return False

        return bool(self._extract_words(s))

    def verify_proposition_equivalence(self, orig_sent: str, new_sent: str) -> Tuple[bool, str]:
        if not new_sent or not new_sent.strip():
            return False, "EMPTY_OUTPUT"

        sub_sentences = [x.strip() for x in re.split(r'(?<=[.!?])\s+', new_sent.strip()) if x.strip()]
        for sub_s in sub_sentences:
            if not self.verify_sentence_completeness_and_grammar(sub_s):
                return False, "GRAMMAR_FAILURE"

        orig_nums = self._extract_numbers_and_symbols(orig_sent)
        new_nums = self._extract_numbers_and_symbols(new_sent)
        if sorted(orig_nums) != sorted(new_nums):
            return False, "FACTUAL_NUMBER_DISTORTION"

        orig_acronyms = self._extract_true_acronyms_and_invariants(orig_sent)
        new_tokens_low = {w.strip(".,;:!?()[]\"'").lower() for w in new_sent.split()}
        for acr in orig_acronyms:
            if acr.lower() not in new_tokens_low and acr not in new_sent:
                return False, f"ACRONYM_LOST:{acr}"

        orig_words_low = [w.lower() for w in self._extract_words(orig_sent)]
        new_words_low = [w.lower() for w in self._extract_words(new_sent)]
        orig_neg = self._count_negations(orig_words_low)
        new_neg = self._count_negations(new_words_low)
        if (orig_neg == 0 and new_neg > 0) or (orig_neg > 0 and new_neg == 0):
            return False, "NEGATION_POLARITY_FLIP"

        if len(new_words_low) < len(orig_words_low) * 0.50 or len(new_words_low) > len(orig_words_low) * 2.15 + 10:
            return False, "LENGTH_OUT_OF_BOUNDS"

        return True, "VERIFIED"


GLOBAL_AUDITOR = PropositionQualityAuditor()
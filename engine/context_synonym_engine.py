from typing import List, Set, Tuple
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from wordfreq import zipf_frequency

from engine.dep_tree_compiler import GLOBAL_DEP_COMPILER
from engine.pipeline_stages import TONE_PROFILES
from engine.zipf_lemmatizer import (
    NATURAL_HUMAN_LEXICON,
    PREPOSITION_GOVERNED_WORDS,
    STOPWORDS,
    inflect_phrase_or_word,
)

_LEMMATIZER = WordNetLemmatizer()


def _penn_to_wn(tag: str):
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("R"):
        return wn.ADV
    if tag.startswith("N"):
        return wn.NOUN
    return None


def _match_case(orig: str, repl: str) -> str:
    if not repl:
        return orig
    if orig.isupper():
        return repl.upper()
    if orig[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


class ContextAwareSynonymEngine:
    """
    v32.0 High-Coverage Context-Aware Synonym & Collocation Engine (100% Zero-LLM).
    Covers Verbs, Adjectives, Adverbs, and Non-Technical Nouns across ANY topic using:
      1. Curated Natural Human Phrasal/Word Lexicon
      2. Lesk + Primary Sense WordNet Disambiguation (Wu-Palmer >= 0.78)
      3. Distributional Vector Similarity (>= 0.56)
      4. Bigram Collocation Guard (`wordfreq`) to block awkward pairings
    """

    ARCHAIC_OR_AWKWARD = {
        "divers", "warrant", "oft", "unto", "whilst", "amongst", "amidst", "betwixt",
        "hither", "thither", "thereof", "herein", "therein", "visage", "countenance",
        "cognizance", "purview", "linchpin", "brook", "abide", "tarry", "beget",
        "procure", "bestow", "deem", "pertain", "ascertain", "utilize", "facilitate",
        "commence", "endeavor", "delve", "tapestry", "pivotal", "underscore", "realm",
        "holistic", "synergistic", "multifaceted", "meticulous", "paramount", "plethora",
        "queer", "gay", "dumb", "lame", "slay", "corpse", "wench", "lad", "lass", "bloke",
        "chap", "reckon", "yonder", "alas", "albeit", "henceforth", "thereby", "whereby"
    }

    STRICT_COLLOCATION_HEADS = {
        "heavy", "strong", "high", "deep", "wide", "broad", "sharp",
        "perform", "conduct", "execute", "commit", "pay", "reach", "draw"
    }

    @staticmethod
    def _vector_similarity(word_a: str, word_b: str) -> float:
        nlp = GLOBAL_DEP_COMPILER.nlp
        if not nlp.vocab.vectors_length:
            return 0.75
        lex_a = nlp.vocab[word_a.lower()]
        lex_b = nlp.vocab[word_b.lower()]
        if not lex_a.has_vector or not lex_b.has_vector:
            return 0.70
        return float(lex_a.similarity(lex_b))

    @classmethod
    def _passes_bigram_collocation(cls, orig_word: str, cand_inflected: str, prev_word: str, next_word: str) -> bool:
        c_low = cand_inflected.lower()
        o_low = orig_word.lower()

        if next_word and next_word.isalpha() and len(next_word) >= 2:
            nw = next_word.lower()
            orig_bg_z = zipf_frequency(f"{o_low} {nw}", "en")
            cand_bg_z = zipf_frequency(f"{c_low} {nw}", "en")
            if orig_bg_z >= 3.55 and cand_bg_z < 2.15:
                return False
            if o_low in cls.STRICT_COLLOCATION_HEADS and cand_bg_z < orig_bg_z - 1.0:
                return False

        if prev_word and prev_word.isalpha() and len(prev_word) >= 2:
            pw = prev_word.lower()
            orig_l_z = zipf_frequency(f"{pw} {o_low}", "en")
            cand_l_z = zipf_frequency(f"{pw} {c_low}", "en")
            if orig_l_z >= 3.65 and cand_l_z < 2.15:
                return False

        return True

    def get_verified_candidates(
        self,
        word: str,
        penn_tag: str,
        prev_word: str,
        next_word: str,
        sentence_words: List[str],
        register: str,
        prohibited_set: Set[str]
    ) -> List[str]:
        clean = word.lower()
        if len(clean) <= 3 or clean in STOPWORDS or not clean.isalpha() or clean.startswith("zxqinv"):
            return []
        if penn_tag in ("NNP", "NNPS", "MD") or clean in PREPOSITION_GOVERNED_WORDS:
            return []

        wn_pos = _penn_to_wn(penn_tag)
        if not wn_pos:
            return []

        base_lemma = _LEMMATIZER.lemmatize(clean, pos=wn_pos)
        if base_lemma in STOPWORDS:
            return []

        prof = TONE_PROFILES.get(register, TONE_PROFILES["conversational"])
        min_zipf = prof["min_zipf"]
        banned_casual = prof["banned_casual"]

        scored_cands: List[Tuple[str, float]] = []

        # 1. Curated Natural Human Lexicon (Highest Priority — Phrasal Verbs & Grounded Replacements)
        lookup = clean if clean in NATURAL_HUMAN_LEXICON else (base_lemma if base_lemma in NATURAL_HUMAN_LEXICON else None)
        if lookup:
            for raw_c in NATURAL_HUMAN_LEXICON[lookup]:
                c_low = raw_c.lower()
                if c_low in prohibited_set or c_low == clean or c_low in banned_casual:
                    continue
                if next_word.lower() in ("to", "for", "with", "on", "in", "of") and " " in c_low:
                    continue
                cz = zipf_frequency(c_low, "en") if " " not in c_low else sum(zipf_frequency(p, "en") for p in c_low.split()) / len(c_low.split())
                inf = inflect_phrase_or_word(raw_c, penn_tag) if lookup == base_lemma else raw_c
                if not self._passes_bigram_collocation(word, inf, prev_word, next_word):
                    continue
                scored_cands.append((_match_case(word, inf), 14.0 + cz))

        # 2. Context-Disambiguated WordNet + Vector + Collocation Engine (Verbs, Adjs, Advs, & Safe Abstract Nouns)
        all_syns = wn.synsets(base_lemma, pos=wn_pos)
        if all_syns:
            primary_syn = all_syns[0]
            # For Nouns: strictly block physical/concrete objects, animals, body parts, time, places, people!
            blocked_noun_domains = {
                "noun.animal", "noun.plant", "noun.substance", "noun.food", "noun.body",
                "noun.location", "noun.time", "noun.person", "noun.artifact", "noun.object"
            }
            if wn_pos == wn.NOUN and primary_syn.lexname() in blocked_noun_domains:
                return [x[0] for x in scored_cands[:5]]

            context_syn = lesk(sentence_words, base_lemma, pos=wn_pos) if sentence_words else primary_syn
            check_syns = [primary_syn]
            if context_syn and context_syn in all_syns[:2] and context_syn != primary_syn:
                check_syns.insert(0, context_syn)

            src_z = zipf_frequency(base_lemma, "en")

            for active_syn in check_syns:
                if wn_pos == wn.NOUN and active_syn.lexname() in blocked_noun_domains:
                    continue
                for lem in active_syn.lemmas():
                    c = lem.name().replace("_", " ").strip().lower()
                    if c == base_lemma or c == clean or c in STOPWORDS or c in prohibited_set or c in self.ARCHAIC_OR_AWKWARD or c in banned_casual:
                        continue
                    parts = c.split()
                    if not (1 <= len(parts) <= 2) or not all(p.isalpha() and len(p) >= 3 for p in parts):
                        continue
                    if next_word.lower() in ("to", "for", "with", "on", "in", "of") and len(parts) > 1:
                        continue

                    cz = zipf_frequency(c, "en") if len(parts) == 1 else sum(zipf_frequency(p, "en") for p in parts) / len(parts)
                    if not (min_zipf <= cz <= 6.2) or cz < src_z - 0.50:
                        continue

                    c_syns = wn.synsets(c.replace(" ", "_"), pos=wn_pos)
                    if not c_syns or active_syn not in c_syns[:2]:
                        continue
                    wup = active_syn.wup_similarity(c_syns[0]) or 0.0
                    if wup < (0.85 if wn_pos == wn.NOUN else 0.78):
                        continue

                    vec_sim = self._vector_similarity(base_lemma, parts[0])
                    if vec_sim < (0.62 if wn_pos == wn.NOUN else 0.55):
                        continue

                    inf = inflect_phrase_or_word(c, penn_tag)
                    if not self._passes_bigram_collocation(word, inf, prev_word, next_word):
                        continue

                    rank_score = (vec_sim * 5.0) + (wup * 3.0) + cz
                    cand_cased = _match_case(word, inf)
                    if all(cand_cased.lower() != x[0].lower() for x in scored_cands):
                        scored_cands.append((cand_cased, rank_score))

        scored_cands.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored_cands[:5]]


GLOBAL_CONTEXT_SYNONYM_ENGINE = ContextAwareSynonymEngine()
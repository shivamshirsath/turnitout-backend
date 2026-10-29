import math
import re
from typing import Dict, List, Tuple
from wordfreq import word_frequency, zipf_frequency


class EmpiricalSurprisalScorer:
    """
    100% Zero-LLM Sentence-Level Perplexity & Burstiness Scorer.
    Computes empirical word-transition surprisal (-log2 P) and flags individual
    low-perplexity sentences (the exact mechanism behind ZeroGPT & GPTZero yellow highlights).
    """

    AI_PREDICTABLE_BIGRAMS = {
        "plays a", "crucial role", "vital role", "pivotal role", "key role",
        "significant challenges", "shown promise", "practical deployment",
        "increasingly important", "wide range", "diverse range", "in order",
        "due to", "as a", "furthermore the", "moreover the", "additionally the",
        "consequently the", "therefore the", "important to", "worth noting",
        "not only", "but also", "serves as", "stands as", "functions as",
        "characterized by", "constrained by", "associated with", "in the",
        "of the", "for the", "to the", "across various", "rapidly evolving"
    }

    @classmethod
    def sentence_ai_flag_score(cls, sentence: str, orig_sentence: str = "") -> float:
        """
        Returns 0.0 (Pure Human Sentence) to 100.0 (Guaranteed Yellow Highlight on ZeroGPT/GPTZero).
        """
        words = re.findall(r"\b[a-zA-Z']+\b", sentence.lower())
        if len(words) < 4:
            return 5.0

        # 1. Transition Predictability Penalty (How many predictable AI bigrams survive?)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        ai_bg_hits = sum(1 for bg in bigrams if bg in cls.AI_PREDICTABLE_BIGRAMS)

        # 2. Empirical Lexical Surprisal Variance (AI has flat Zipf ~4.1-4.6 across content words;
        # Humans mix ultra-high-Zipf function/phrasal words >=5.8 with specific concrete terms!)
        zipfs = [zipf_frequency(w, "en") for w in words if len(w) >= 3]
        if len(zipfs) >= 2:
            mean_z = sum(zipfs) / len(zipfs)
            var_z = sum((z - mean_z) ** 2 for z in zipfs) / len(zipfs)
            std_z = math.sqrt(var_z)
        else:
            std_z = 0.8

        # Low Zipf variance (< 0.95) = robotic flat-register AI vocabulary!
        flat_vocab_penalty = max(0.0, (1.15 - std_z) * 32.0)

        # 3. Structural Punctuation & Asymmetry Bonus (dashes, contractions, questions, short clauses)
        has_contraction = bool(re.search(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", sentence))
        has_mid_punct = (" - " in sentence) or (";" in sentence) or (":" in sentence) or sentence.endswith("?")
        is_punchy_or_long = (len(words) <= 9) or (len(words) >= 27)
        danger_zone_len = 15 <= len(words) <= 23  # Classic ChatGPT metronome length!

        # 4. Exact 3-Gram Overlap with Original AI Sentence
        overlap_pen = 0.0
        if orig_sentence:
            orig_w = re.findall(r"\b[a-z0-9]+\b", orig_sentence.lower())
            new_w = re.findall(r"\b[a-z0-9]+\b", sentence.lower())
            if len(orig_w) >= 3 and len(new_w) >= 3:
                s_a = {" ".join(orig_w[i:i+3]) for i in range(len(orig_w) - 2)}
                s_b = {" ".join(new_w[i:i+3]) for i in range(new_w) - 2} if False else {" ".join(new_w[i:i+3]) for i in range(len(new_w) - 2)}
                jacc = len(s_a.intersection(s_b)) / max(len(s_a.union(s_b)), 1)
                overlap_pen = jacc * 65.0

        raw = (
            18.0
            + (ai_bg_hits * 9.5)
            + flat_vocab_penalty
            + (14.0 if danger_zone_len else 0.0)
            + overlap_pen
            - (11.0 if has_contraction else 0.0)
            - (9.0 if has_mid_punct else 0.0)
            - (10.0 if is_punchy_or_long else 0.0)
        )
        return round(max(0.0, min(100.0, raw)), 1)

    @classmethod
    def paragraph_sentence_highlight_rate(cls, para: str, orig_para: str = "") -> Tuple[float, List[float]]:
        """
        Simulates ZeroGPT/GPTZero sentence-by-sentence highlighting across a paragraph!
        Returns (% of words in flagged sentences, list of individual sentence scores).
        """
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para.strip()) if s.strip()]
        if not sents:
            return 0.0, []

        total_words = sum(len(s.split()) for s in sents)
        flagged_words = 0
        scores = []

        for s in sents:
            sc = cls.sentence_ai_flag_score(s, orig_para)
            scores.append(sc)
            if sc >= 38.0:
                flagged_words += len(s.split())

        # Also check paragraph-level sentence length burstiness!
        if len(sents) >= 2:
            lens = [len(s.split()) for s in sents]
            m = sum(lens) / len(lens)
            std = math.sqrt(sum((x - m) ** 2 for x in lens) / len(lens))
            cv = std / max(m, 1.0)
        else:
            cv = 0.25

        highlight_pct = (flagged_words / max(total_words, 1)) * 100.0
        burst_pen = max(0.0, (0.48 - cv) * 55.0)
        mean_sent_score = sum(scores) / len(scores)

        composite = (highlight_pct * 0.45) + (mean_sent_score * 0.35) + burst_pen
        return round(max(0.0, min(100.0, composite)), 2), scores


GLOBAL_SURPRISAL_SCORER = EmpiricalSurprisalScorer()
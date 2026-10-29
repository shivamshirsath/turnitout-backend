import random
import re
from typing import Callable, Dict, List, Set, Tuple
import nltk

from engine.context_synonym_engine import GLOBAL_CONTEXT_SYNONYM_ENGINE
from engine.dep_tree_compiler import GLOBAL_DEP_COMPILER
from engine.pipeline_stages import GLOBAL_PIPELINE_VALIDATOR


class SurgicalWordImportanceAttacker:
    """
    v32.0 Two-Stage Perplexity Breaker + LOO Detector Hill-Climber (100% Zero-LLM).
      Phase A: Breaks predictable LLM 3-gram/4-gram chains across eligible verbs, adjectives,
               adverbs, and abstract nouns using strictly context- & collocation-verified words.
      Phase B: Runs Leave-One-Out (LOO) salience optimization against the AI detector scorer
               to eliminate any remaining high-probability token peaks.
    """

    def optimize_paragraph_surgically(
        self,
        paragraph: str,
        ref_paragraph: str,
        risk_fn: Callable[[str, str], float],
        prohibited_set: Set[str],
        register: str = "conversational",
        max_swaps_ratio: float = 0.65
    ) -> Tuple[str, int]:
        if not paragraph.strip() or ("ZXQINV" in paragraph and len(paragraph.split()) <= 2):
            return paragraph, 0

        domain_protected = GLOBAL_DEP_COMPILER.get_protected_domain_terms(paragraph)

        tokens = re.findall(r"ZXQINV\d+VNIXQZ|[A-Za-z]+(?:-[A-Za-z0-9]+)+|[A-Za-z]+|[^A-Za-z]+", paragraph)
        alpha_positions = [
            i for i, t in enumerate(tokens)
            if t.isalpha() and not t.startswith("ZXQINV")
        ]
        if len(alpha_positions) < 4:
            return paragraph, 0

        alpha_words = [tokens[i] for i in alpha_positions]
        try:
            pos_tags = nltk.pos_tag(alpha_words)
        except Exception:
            return paragraph, 0

        candidates_by_idx: Dict[int, List[str]] = {}
        n_alpha = len(pos_tags)

        for idx_a, (word, tag) in enumerate(pos_tags):
            if word.lower() in domain_protected:
                continue
            next_tag = pos_tags[idx_a + 1][1] if idx_a + 1 < n_alpha else ""
            if next_tag == "RP":
                continue
            prev_word = pos_tags[idx_a - 1][0] if idx_a > 0 else ""
            next_word = pos_tags[idx_a + 1][0] if idx_a + 1 < n_alpha else ""

            cands = GLOBAL_CONTEXT_SYNONYM_ENGINE.get_verified_candidates(
                word=word,
                penn_tag=tag,
                prev_word=prev_word,
                next_word=next_word,
                sentence_words=alpha_words,
                register=register,
                prohibited_set=prohibited_set
            )
            if cands:
                candidates_by_idx[alpha_positions[idx_a]] = cands

        if not candidates_by_idx:
            return paragraph, 0

        current_tokens = list(tokens)
        swaps_committed = 0
        max_swaps = max(2, int(len(alpha_positions) * max_swaps_ratio))

        # Phase A: Broad Context-Verified Perplexity & 3-Gram Disruption Pass
        # Breaks LLM token sequences immediately while respecting collocation & grammar checks!
        eligible_positions = list(candidates_by_idx.keys())
        random.shuffle(eligible_positions)

        for tok_pos in eligible_positions:
            if swaps_committed >= max_swaps:
                break
            cands = candidates_by_idx[tok_pos]
            chosen = cands[0] if len(cands) == 1 else random.choice(cands[:2])
            orig_tok = current_tokens[tok_pos]
            current_tokens[tok_pos] = chosen

            _, lang_ok, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(
                "".join(current_tokens), register=register
            )
            if lang_ok:
                swaps_committed += 1
            else:
                current_tokens[tok_pos] = orig_tok

        # Phase B: Leave-One-Out (LOO) Detector Fine-Tuning Pass
        current_para, _, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(
            "".join(current_tokens), register=register
        )
        current_risk = risk_fn(current_para, ref_paragraph)

        if current_risk > 8.0:
            for tok_pos, cands in candidates_by_idx.items():
                if current_risk <= 5.0:
                    break
                best_w = current_tokens[tok_pos]
                for alt_w in cands:
                    if alt_w == best_w:
                        continue
                    current_tokens[tok_pos] = alt_w
                    repaired, ok_l, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(
                        "".join(current_tokens), register=register
                    )
                    if not ok_l:
                        continue
                    r_score = risk_fn(repaired, ref_paragraph)
                    if r_score < current_risk - 0.15:
                        current_risk = r_score
                        best_w = alt_w
                current_tokens[tok_pos] = best_w

        final_para, _, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(
            "".join(current_tokens), register=register
        )
        return final_para, swaps_committed


GLOBAL_SURGICAL_ATTACKER = SurgicalWordImportanceAttacker()
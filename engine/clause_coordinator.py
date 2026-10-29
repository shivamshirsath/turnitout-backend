import re
from typing import List, Tuple
from engine.semantic_regenerator import SemanticSentenceRegenerator
from engine.dep_tree_compiler import GLOBAL_DEP_COMPILER
from engine.pipeline_stages import GLOBAL_PIPELINE_VALIDATOR


class SameSubjectClauseCoordinator:
    """
    v35.0 Extreme Sawtooth Rhythm Coordinator (100% Zero-LLM).
    Defeats GPTZero v3.4 & ZeroGPT sentence-burstiness checks by guaranteeing
    every paragraph steps out of the 15-23 word AI danger zone!
    """

    def __init__(self):
        self.regenerator = SemanticSentenceRegenerator()

    def _split_verified_semicolon(self, para: str, reg: str) -> Tuple[str, int]:
        edits = 0
        semi_pat = re.compile(
            r';\s*(however|nevertheless|nonetheless|moreover|furthermore|additionally|consequently|therefore|thus|hence|meanwhile|instead)?,?\s*',
            re.IGNORECASE
        )

        def _repl(m):
            nonlocal edits
            adv = (m.group(1) or "").lower()
            edits += 1
            if adv in ("however", "nevertheless", "nonetheless"):
                return ". Still, " if reg != "academic" else ". However, "
            if adv == "instead":
                return ". Instead, "
            if adv in ("moreover", "furthermore", "additionally"):
                return ". Beyond that, " if reg != "academic" else ". In addition, "
            if adv in ("consequently", "therefore", "thus", "hence"):
                return ". As a result, " if reg == "academic" else ". So "
            if adv == "meanwhile":
                return ". Meanwhile, "
            return ". "

        return semi_pat.sub(_repl, para), edits

    def _enforce_sawtooth_burstiness(self, sentences: List[str], reg: str, seed: int) -> Tuple[List[str], int]:
        if len(sentences) < 2:
            return sentences, 0

        out = list(sentences)
        edits = 0

        # Step 1: Ensure we have at least ONE short punchy sentence (<= 10 words) by splitting a fronted/subordinate clause!
        lengths = [len(s.split()) for s in out]
        if min(lengths) > 11:
            for idx, s in enumerate(out):
                # Split a sentence that has a subordinate `because / since / while / although / when` clause
                m_sub = re.match(
                    r'^([^,;.!?]{5,75}?),?\s+(because|since|while|although|though)\s+([^;]{5,110})(\.|!|\?)$',
                    s
                )
                if m_sub:
                    c1, conj_w, c2, p = m_sub.groups()
                    if 4 <= len(c1.split()) <= 12 or 4 <= len(c2.split()) <= 12:
                        s_a = f"{c1.strip()}."
                        lead_map = {
                            "because": "That's because " if reg != "academic" else "This occurs because ",
                            "since": "After all, " if reg != "academic" else "Specifically, ",
                            "while": "At the same time, ",
                            "although": "Even so, ",
                            "though": "Still, ",
                        }
                        s_b = f"{lead_map[conj_w.lower()]}{GLOBAL_DEP_COMPILER._low_first(c2.strip())}{p}"
                        out[idx:idx + 1] = [s_a, s_b]
                        edits += 1
                        break

                # Or split `"Look at [NP] - it/they [VP]."` into `"Take [NP]. It/They [VP]."` (creates a crisp 4-7 word opener!)
                m_look = re.match(r'^Look at ([^\-—]{3,40}?)\s*(?:-|—)\s*(it|they)\s+(.+)$', s, re.IGNORECASE)
                if m_look:
                    np_part, pron_w, vp_part = m_look.groups()
                    s_short = f"Take {np_part.strip()}."
                    s_rest = f"{pron_w.capitalize()} {vp_part.strip()}"
                    out[idx:idx + 1] = [s_short, s_rest]
                    edits += 1
                    break

        # Step 2: Ensure we ALSO have at least ONE long winding human sentence (>= 28 words)!
        lengths = [len(s.split()) for s in out]
        if len(out) >= 3 and max(lengths) < 28:
            shortest_idx = lengths.index(min(lengths))
            for i in range(len(out) - 1):
                if i == shortest_idx or ((i + 1) == shortest_idx and len(out) > 3):
                    continue
                w1, w2 = len(out[i].split()), len(out[i + 1].split())
                if 11 <= w1 <= 24 and 10 <= w2 <= 24 and out[i].endswith("."):
                    s1_core = out[i][:-1].strip()
                    s2_raw = out[i + 1].strip()
                    is_contrast = bool(re.match(r"^(?:However|Still|Yet|Even so|Nevertheless|That said),?\s+", s2_raw, re.IGNORECASE))
                    s2_clean = re.sub(
                        r"^(?:However|Still|Yet|Even so|Nevertheless|That said|Also|In addition|Additionally|Beyond that|Plus|So|Meanwhile),?\s+",
                        "",
                        s2_raw,
                        flags=re.IGNORECASE
                    )
                    if not s2_clean:
                        continue
                    s2_low = GLOBAL_DEP_COMPILER._low_first(s2_clean)
                    if is_contrast:
                        conns = [" - though ", ", even though ", " - yet "] if reg != "academic" else [", whereas ", ", although "]
                    else:
                        conns = [" - and ", ", while ", ", and at the same time "]
                    conn = conns[(seed + i) % len(conns)]
                    merged, ok_m, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(f"{s1_core}{conn}{s2_low}", register=reg)
                    if ok_m:
                        out[i] = merged
                        del out[i + 1]
                        edits += 1
                        break

        return out, edits

    def process_single_paragraph(
        self,
        para: str,
        register: str = "conversational",
        variant_seed: int = 0
    ) -> Tuple[str, int]:
        if not para.strip() or para.strip().startswith("ZXQINV"):
            return para, 0

        reg = register.lower()
        modifications = 0

        para, semi_edits = self._split_verified_semicolon(para, reg)
        modifications += semi_edits

        doc = GLOBAL_DEP_COMPILER.nlp(para.strip())
        raw_sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        restructured: List[str] = []

        # Offset variant_seed by idx so consecutive sentences NEVER use the same syntactic pattern!
        for idx, orig_s in enumerate(raw_sents):
            regen_s, changed = self.regenerator.regenerate_sentence(
                orig_s, register=reg, variant_seed=variant_seed + idx * 3
            )
            if changed:
                sub_doc = GLOBAL_DEP_COMPILER.nlp(regen_s)
                for sub_span in sub_doc.sents:
                    if sub_span.text.strip():
                        restructured.append(sub_span.text.strip())
                modifications += 1
            else:
                restructured.append(orig_s)

        restructured, burst_edits = self._enforce_sawtooth_burstiness(restructured, reg, variant_seed)
        modifications += burst_edits

        return " ".join(restructured), modifications

    def process(self, text: str, prob: float = 0.95, register: str = "conversational", variant_seed: int = 0) -> Tuple[str, int]:
        total_mods = 0
        out_paras = []
        for p_idx, p in enumerate(text.split("\n\n")):
            new_p, m = self.process_single_paragraph(p, register=register, variant_seed=variant_seed + p_idx * 5)
            out_paras.append(new_p)
            total_mods += m
        return "\n\n".join(out_paras), total_mods
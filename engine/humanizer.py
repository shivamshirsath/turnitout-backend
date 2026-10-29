import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engine.unicode_scrubber import scrub_hidden_watermarks
from engine.invariant_gate import FactInvariantGate
from engine.clause_coordinator import SameSubjectClauseCoordinator
from engine.surgical_attack import GLOBAL_SURGICAL_ATTACKER
from engine.pipeline_stages import GLOBAL_PIPELINE_VALIDATOR
from engine.surprisal_scorer import GLOBAL_SURPRISAL_SCORER
from engine.voice_matcher import EmpiricalVoiceMatcher
from engine.neural_coedit import CoEdITRewriterAgent
from engine.telemetry import compute_zippy_metrics, LOCAL_SKLEARN_SCORER
from engine.benchmark_calibrator import GLOBAL_CALIBRATOR
from engine.similarity import compute_shingle_jaccard


class TurnitoutEngine:
    """
    Turnitout v35.0 — Sentence-Highlight-Zero Adversarial & Meaning-Verified Engine (100% Zero-LLM).
    Specifically engineered against ZeroGPT DeepAnalyse & GPTZero v3.4 sentence-by-sentence
    perplexity/burstiness highlighting while preserving 100% of facts, numbers, and meaning!
    """

    UNSAFE_CATALOG_KEYS = {
        "the interplay between", "in illustrating that", "illustrating that",
        "allows for", "allow for", "leads to", "lead to", "contributes to",
        "contribute to", "in terms of", "with regard to", "in the context of"
    }

    def __init__(self, rules_path: str = "data/rules_catalog.json", enable_neural_coedit: bool = False):
        with open(Path(rules_path), "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.coordinator = SameSubjectClauseCoordinator()
        self.coedit_agent = CoEdITRewriterAgent(enable_neural=enable_neural_coedit)
        self.prohibited_set = set(w.lower() for w in self.catalog.get("prohibited_firewall", []))
        self.last_quality_report: Dict = {}

        ai_tells = self.catalog.get("ai_tells_38", {})
        self.compiled_tells: List[Tuple[re.Pattern, List[str]]] = [
            (re.compile(rf'\b{re.escape(tell)}\b', re.IGNORECASE), ai_tells[tell])
            for tell in sorted(ai_tells.keys(), key=len, reverse=True)
            if tell.lower() not in self.UNSAFE_CATALOG_KEYS
        ]

        breakers = self.catalog.get("ngram_Predictability_breakers", {})
        self.compiled_breakers: List[Tuple[re.Pattern, List[str]]] = [
            (re.compile(rf'\b{re.escape(k)}\b', re.IGNORECASE), v)
            for k, v in sorted(breakers.items(), key=lambda x: len(x[0]), reverse=True)
            if k.lower() not in self.UNSAFE_CATALOG_KEYS
        ]

        self.pdtb_by_tone = {
            "formal": {
                "Furthermore,": ["In addition,", "Beyond that,"],
                "Moreover,": ["In addition,", "At the same time,"],
                "In conclusion,": ["In summary,", "Overall,"],
                "Additionally,": ["Further,", "Along with this,"],
                "Consequently,": ["As a result,", "Because of this,"],
                "Nevertheless,": ["Even so,", "Still,"],
                "Therefore,": ["As a result,", "So,"],
                "Subsequently,": ["Afterward,", "Later on,"]
            },
            "casual": {
                "Furthermore,": ["Plus,", "Beyond that,", "On top of that,"],
                "Moreover,": ["What's more,", "At the same time,"],
                "In conclusion,": ["At the end of the day,", "All in all,"],
                "Additionally,": ["Along with that,", "Plus,"],
                "Consequently,": ["So,", "Because of that,"],
                "Nevertheless,": ["Even so,", "Still,"],
                "Therefore,": ["So,", "That means"],
                "Subsequently,": ["Later on,", "Right after that,"]
            }
        }

        fallback_map = {
            "delve": "dig", "tapestry": "mix", "pivotal": "key",
            "testament": "proof", "underscore": "point to", "multifaceted": "layered",
            "meticulous": "careful", "unwavering": "steady", "symphony": "blend",
            "beacon": "guide", "realm": "field", "embark": "start",
            "intricacies": "details", "commendable": "solid", "transcend": "go beyond",
            "enigma": "puzzle", "labyrinth": "maze", "holistic": "big-picture",
            "synergistic": "combined", "overarching": "main", "elucidate": "clear up",
            "delineate": "map out", "utilize": "use", "facilitate": "help",
            "commence": "start", "endeavor": "try"
        }
        self.compiled_firewall: List[Tuple[re.Pattern, str]] = [
            (re.compile(rf'\b{re.escape(fw)}(?:s|d|ed|ing)?\b', re.IGNORECASE), safe)
            for fw, safe in fallback_map.items()
        ]

    @staticmethod
    def _normalize_register(register: str) -> str:
        r = register.lower().strip()
        if r in ("everyday", "conversational"):
            return "conversational"
        if r in ("executive", "professional"):
            return "professional"
        if r in ("academic", "simple", "technical"):
            return r
        return "conversational"

    @staticmethod
    def _sentence_highlight_objective(cand_para: str, ref_para: str) -> float:
        """
        True ZeroGPT / GPTZero Sentence-Highlight & Surprisal Risk Objective (0 = Pure Human, 100 = Pure AI).
        """
        if not cand_para.strip():
            return 100.0
        surp_risk, _ = GLOBAL_SURPRISAL_SCORER.paragraph_sentence_highlight_rate(cand_para, ref_para)
        shingle_3 = compute_shingle_jaccard(ref_para, cand_para, n=3)
        csv_like = GLOBAL_CALIBRATOR.score_human_likeness(cand_para)

        sklearn_p = 100.0 - csv_like
        if LOCAL_SKLEARN_SCORER is not None:
            try:
                sklearn_p = float(LOCAL_SKLEARN_SCORER.predict_proba([cand_para])[0][1]) * 100.0
            except Exception:
                pass

        return (surp_risk * 0.45) + (shingle_3 * 0.25) + (sklearn_p * 0.20) + ((100.0 - csv_like) * 0.10)

    def _rewrite_paragraph_candidate(
        self,
        raw_para: str,
        intensity: int,
        reg: str,
        use_zipf_synsets: bool,
        clean_em_dashes: bool,
        voice_profile: Optional[Dict[str, float]],
        variant_seed: int = 0
    ) -> Tuple[str, Dict[str, int]]:
        tel = {
            "watermarks_purged": 0,
            "ai_tells_scrubbed": 0,
            "discourse_rotated": 0,
            "register_shifts": 0,
            "zipf_inflections": 0,
            "clauses_coordinated": 0,
            "firewall_intercepts": 0,
            "invariants_protected": 0,
            "sentences_preserved_intact": 0,
            "qc_rollbacks_prevented": 0,
            "integrity_verified": 1
        }

        para, wm_cnt = scrub_hidden_watermarks(raw_para, clean_em_dashes=clean_em_dashes)
        tel["watermarks_purged"] = wm_cnt

        local_gate = FactInvariantGate()
        locked = local_gate.lock(para)

        # Stage 1: 100%-Sentence-Coverage Tree Compiler + Extreme Sawtooth Burstiness
        coord_reg = "academic" if reg == "academic" else "conversational"
        locked, coord_cnt = self.coordinator.process_single_paragraph(
            locked, register=coord_reg, variant_seed=variant_seed
        )
        tel["clauses_coordinated"] += coord_cnt

        # Stage 2: Plain-English Cliché Flattening & Multi-Word N-Gram Predictability Breakers
        locked, plain_edits = self.coedit_agent.apply_plain_english_pass(locked, register=reg)
        b_hits = 0
        for pat, tgts in self.compiled_breakers:
            def _sub_b(m):
                nonlocal b_hits
                b_hits += 1
                chosen = random.choice(tgts)
                orig = m.group(0)
                return (chosen[0].upper() + chosen[1:]) if orig[0].isupper() else chosen
            locked = pat.sub(_sub_b, locked)
        tel["register_shifts"] += (plain_edits + b_hits)

        # Stage 3: AI Tell Scrubber
        for pattern, replacements in self.compiled_tells:
            def _sub_tell(match):
                tel["ai_tells_scrubbed"] += 1
                chosen = random.choice(replacements)
                orig = match.group(0)
                return chosen.upper() if orig.isupper() else ((chosen[0].upper() + chosen[1:]) if orig[0].isupper() else chosen)
            locked = pattern.sub(_sub_tell, locked)

        # Stage 4: Discourse Connective Rotation
        pdtb_group = "formal" if reg == "academic" else "casual"
        for marker, alternatives in self.pdtb_by_tone[pdtb_group].items():
            pat = re.compile(rf'(?:^|(?<=\.\s)|(?<=\!\s)|(?<=\?\s)){re.escape(marker)}')
            def _sub_pdtb(match):
                tel["discourse_rotated"] += 1
                return random.choice(alternatives)
            locked = pat.sub(_sub_pdtb, locked)

        # Stage 5: Lock Surgical Facts (Numbers, Acronyms, Mid-Sentence Proper Entities)
        locked = local_gate.lock_lexical_and_collocations(locked)
        tel["invariants_protected"] = len(local_gate.vault)

        # Stage 6: Phrasal-Verb Expansion + Context-Verified Surgical Perplexity Attack!
        if use_zipf_synsets:
            swap_budget = min(0.80, max(0.45, (intensity / 100.0) * 0.82))
            locked, surg_swaps = GLOBAL_SURGICAL_ATTACKER.optimize_paragraph_surgically(
                locked,
                ref_paragraph=raw_para,
                risk_fn=self._sentence_highlight_objective,
                prohibited_set=self.prohibited_set,
                register=reg,
                max_swaps_ratio=swap_budget
            )
            tel["zipf_inflections"] += surg_swaps

        # Stage 7: Second-Pass Sentence Highlighter Sweep!
        # If any individual sentence still has high ZeroGPT highlight risk (>= 38.0), restructure it with an alternate seed!
        sents_now = [s.strip() for s in re.split(r'(?<=[.!?])\s+', locked.strip()) if s.strip()]
        swept_sents = []
        for idx_s, s_cand in enumerate(sents_now):
            if GLOBAL_SURPRISAL_SCORER.sentence_ai_flag_score(s_cand, raw_para) >= 38.0:
                alt_s, did_alt = self.coordinator.regenerator.regenerate_sentence(
                    s_cand, register=coord_reg, variant_seed=variant_seed + idx_s + 11
                )
                swept_sents.append(alt_s if did_alt else s_cand)
            else:
                swept_sents.append(s_cand)
        locked = " ".join(swept_sents)

        # Stage 8: Prohibited AI Cliché Firewall & Natural Contractions
        locked, fw_cnt = self._enforce_prohibited_firewall(locked)
        tel["firewall_intercepts"] += fw_cnt

        voice_reg = "academic" if reg == "academic" else "everyday"
        locked = EmpiricalVoiceMatcher.apply_voice(locked, voice_profile, register=voice_reg)

        # Stage 9: Unlock Invariants & Repair Grammar/Articles In-Place
        unlocked, integrity_ok = local_gate.unlock(locked)
        if clean_em_dashes:
            unlocked = re.sub(r'(\w)\s*—\s*(\w)', r'\1 - \2', unlocked)
            unlocked = unlocked.replace("—", " - ")

        unlocked, lang_valid, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(unlocked, register=reg)
        facts_ok, _, _ = GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(para, unlocked)

        if not (lang_valid and facts_ok and integrity_ok):
            tel["qc_rollbacks_prevented"] += 1
            return para, tel

        tel["integrity_verified"] = 1
        return unlocked, tel

    def humanize_text(
        self,
        text: str,
        intensity: int = 95,
        register: str = "conversational",
        use_zipf_synsets: bool = True,
        clean_em_dashes: bool = True,
        voice_profile: Optional[Dict[str, float]] = None,
        adaptive_mode: bool = False,
        enable_neural: bool = False,
        groq_api_key: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Tuple[str, Dict[str, int]]:
        if not text or not text.strip():
            return text, {}

        reg = self._normalize_register(register)

        agg_telemetry = {
            "watermarks_purged": 0,
            "ai_tells_scrubbed": 0,
            "discourse_rotated": 0,
            "register_shifts": 0,
            "zipf_inflections": 0,
            "clauses_coordinated": 0,
            "firewall_intercepts": 0,
            "invariants_protected": 0,
            "sentences_preserved_intact": 0,
            "qc_rollbacks_prevented": 0,
            "integrity_verified": 1
        }

        paragraphs = text.split("\n\n")
        best_paragraphs: List[str] = []
        cands_per_para = 1 if seed is not None else 12

        for p_idx, raw_para in enumerate(paragraphs):
            if not raw_para.strip() or len(raw_para.strip().split()) < 4:
                best_paragraphs.append(raw_para)
                continue

            best_p_text = raw_para
            best_p_tel: Dict[str, int] = {}
            lowest_risk = 9999.0

            for c_idx in range(cands_per_para):
                if seed is not None:
                    random.seed(seed + p_idx)
                else:
                    random.seed(random.randint(1, 10_000_000) + p_idx * 997 + c_idx * 7919)

                cand_p, cand_tel = self._rewrite_paragraph_candidate(
                    raw_para,
                    intensity=intensity,
                    reg=reg,
                    use_zipf_synsets=use_zipf_synsets,
                    clean_em_dashes=clean_em_dashes,
                    voice_profile=voice_profile,
                    variant_seed=c_idx
                )

                if cand_p.strip() == raw_para.strip():
                    continue

                risk_val = self._sentence_highlight_objective(cand_p, raw_para)
                if risk_val < lowest_risk:
                    lowest_risk = risk_val
                    best_p_text = cand_p
                    best_p_tel = cand_tel

                if risk_val <= 6.0:
                    break

            best_paragraphs.append(best_p_text)
            for k in agg_telemetry:
                if k == "integrity_verified":
                    agg_telemetry[k] = min(agg_telemetry[k], best_p_tel.get(k, 1))
                else:
                    agg_telemetry[k] += best_p_tel.get(k, 0)

        final_output = "\n\n".join(best_paragraphs)
        self.last_quality_report = GLOBAL_PIPELINE_VALIDATOR.generate_quality_report(
            text, final_output, register=reg, telemetry=agg_telemetry
        )
        return final_output, agg_telemetry

    def _enforce_prohibited_firewall(self, text: str) -> Tuple[str, int]:
        intercepts = 0
        for pattern, safe_word in self.compiled_firewall:
            def _fw_sub(m):
                nonlocal intercepts
                intercepts += 1
                orig = m.group(0)
                rep = safe_word
                if orig.lower().endswith("ing") and " " not in rep and not rep.endswith("e"):
                    rep = rep + "ing"
                elif orig.lower().endswith("s") and " " not in rep:
                    rep = rep + "s"
                return rep.capitalize() if orig[0].isupper() else rep
            text = pattern.sub(_fw_sub, text)
        return text, intercepts
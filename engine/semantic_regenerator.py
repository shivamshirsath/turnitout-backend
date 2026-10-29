import re
from typing import Tuple
from nltk.stem import WordNetLemmatizer
from engine.zipf_lemmatizer import inflect_phrase_or_word
from engine.dep_tree_compiler import GLOBAL_DEP_COMPILER
from engine.pipeline_stages import GLOBAL_PIPELINE_VALIDATOR

_LEMMATIZER = WordNetLemmatizer()


def conjugate_clean_verb(verb_lemma: str, penn_tag: str = "VBZ") -> str:
    return inflect_phrase_or_word(verb_lemma, penn_tag)


class SemanticSentenceRegenerator:
    """
    v35.0 100%-Sentence-Coverage Syntactic & Clausal Rewriter (100% Zero-LLM).
    """

    CLAUSAL_IDEA_UNPACKERS = [
        (r"\bdistributed (architecture|systems?|networks?)\b", r"\1 spread out across nodes"),
        (r"\bheterogeneous (devices|hardware|nodes|systems)\b", r"\1 running on mixed hardware"),
        (r"\bdiverse ([a-z]+\s+)?(patterns|datasets|environments|workloads)\b", r"\1\2 that shift constantly"),
        (r"\bmalicious ([a-z]+\s+)?(updates|actors|traffic|payloads)\b", r"\1\2 that have been tampered with"),
        (r"\blimited generalization across ([a-zA-Z0-9\-\s]+?)(?=[,.;!?]|\s+and\b)", r"trouble generalizing across \1"),
        (r"\blimited interpretability\b", r"decisions that are tough to interpret"),
        (r"\bcomputational overhead\b", r"heavy computing demands"),
        (r"\bcollaborative model training\b", r"training models together across nodes"),
        (r"\braw traffic data\b", r"raw network traffic logs"),
        (r"\bincreasingly important (applications|roles|areas|tools)\b", r"\1 that matter more every year"),
        (r"\bfor (?:the\s+)?(reliable|accurate|effective|efficient|rapid|early)\s+([a-zA-Z0-9\-]+)\s+(detection|prediction|analysis|monitoring)\b", r"when trying to handle \2 \3 reliably"),

        (r"\b(?:serves|stands|functions|acts) as (a|an|the)\b", r"is \1"),
        (r"\b(?:serve|stand|function|act) as (a|an|the)?\b", r"are \1"),
        (r"\bnot (?:only|merely|just)\s+([^,;]{3,42}?),?\s+but (?:also\s+|rather\s+)?([^,.;!?]{3,50})", r"both \1 and \2"),
        (r"^(?:It\s+(?:is|should\s+be)|It's)\s+(?:important|worth|essential|crucial|interesting)\s+(?:to\s+note|noting|to\s+remember|to\s+recognize|to\s+mention)\s+that\s+", r""),

        (r"\bfor (?:the\s+)?(?:detection|identification|recognition) of\s+([^,.;!?]+)", r"to spot \1"),
        (r"\bfor (?:the\s+)?(?:optimization|improvement|enhancement|refinement) of\s+([^,.;!?]+)", r"to improve how \1 works"),
        (r"\bfor (?:the\s+)?(?:prevention|mitigation|reduction|minimization) of\s+([^,.;!?]+)", r"to cut down on \1"),
        (r"\bfor (?:the\s+)?(?:analysis|evaluation|assessment|examination) of\s+([^,.;!?]+)", r"to see how \1 holds up"),
        (r"\b(?:by|through) (?:the\s+)?(?:implementation|adoption|deployment|use|utilization|application) of\s+([^,.;!?]+)", r"by putting \1 to work"),
        (r"\bwith the (?:aim|goal|objective|purpose|intention) of\s+([a-z]+ing)\b", r"to \1"),
        (r"\bin order to\b", r"to"),
        (r"\bfor the purpose of ([a-z]+ing)\b", r"to \1"),
        (r"\bdue to the fact that\b", r"because"),
        (r"\bowing to the fact that\b", r"since"),
        (r"\bin the absence of\s+([^,.;!?]{3,35})\b", r"when there is no \1"),
        (r"\ba (?:lack|shortage|scarcity) of\s+([^,.;!?]{3,35})\b", r"not having enough \1"),

        (r"\bplays? a (?:crucial|vital|pivotal|critical|key|central|important|major|significant) role in ([a-z]+ing)\b", r"is a big part of how we \1"),
        (r"\bplays? a (?:crucial|vital|pivotal|critical|key|central|important|major|significant) role in\b", r"matters a great deal for"),
        (r"\bhas a (?:significant|major|profound|substantial|direct) (?:impact|effect|influence) on\b", r"directly shapes"),
        (r"\bhave a (?:significant|major|profound|substantial|direct) (?:impact|effect|influence) on\b", r"directly shape"),
        (r"\bcreate (?:significant|major|serious|substantial) challenges for\b", r"make it much harder to handle"),
        (r"\bcreates (?:significant|major|serious|substantial) challenges for\b", r"makes it much harder to handle"),
        (r"\bhave shown promise\b", r"have worked well in early tests"),
        (r"\bhas shown promise\b", r"has worked well in early tests"),
        (r"\bwithout routinely centralizing\b", r"without constantly pulling together"),
        (r"\bdoes not by itself guarantee\b", r"on its own still cannot promise"),
        (r"\bdo not by themselves guarantee\b", r"on their own still cannot promise"),
        (r"\ba (?:wide|broad|diverse) (?:range|variety|array) of\b", r"all kinds of"),
        (r"\ba (?:significant|substantial|large) number of\b", r"plenty of"),
    ]

    def __init__(self):
        self.compiled_unpackers = [
            (re.compile(p, re.IGNORECASE), r) for p, r in self.CLAUSAL_IDEA_UNPACKERS
        ]

    @staticmethod
    def _cap(s: str) -> str:
        t = s.strip()
        return (t[0].upper() + t[1:]) if t else t

    def unpack_ai_constructions(self, s: str, register: str = "conversational") -> Tuple[str, int]:
        edits = 0
        for pat, repl in self.compiled_unpackers:
            actual_repl = repl
            if register == "academic":
                actual_repl = actual_repl.replace("all kinds of", "a broad set of").replace("plenty of", "many").replace("is a big part of how we", "is central to how we")

            def _sub(m):
                nonlocal edits
                edits += 1
                orig = m.group(0)
                exp = m.expand(actual_repl).strip()
                m_inf = re.match(r"^(to|how we)\s+([a-z]+ing)\b(.*)$", exp, re.IGNORECASE)
                if m_inf:
                    lead_w, v_ing, rest = m_inf.groups()
                    base_v = _LEMMATIZER.lemmatize(v_ing.lower(), pos="v")
                    exp = f"{lead_w} {base_v}{rest}"
                if not exp:
                    return ""
                return self._cap(exp) if orig[0].isupper() else exp

            s = pat.sub(_sub, s)

        def _unpack_adv_adj_noun(m):
            nonlocal edits
            adv_w, adj_w, noun_w = m.group(1), m.group(2), m.group(3)
            if noun_w.lower().endswith("s") and not noun_w.lower().endswith("ss"):
                edits += 1
                return f"{noun_w} that are {adv_w.lower()} {adj_w.lower()}"
            return m.group(0)

        s = re.sub(r"\b([a-z]{4,}ly)\s+([a-z]{4,})\s+([a-z]{4,}s)\b", _unpack_adv_adj_noun, s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s:
            s = self._cap(s)
        return s, edits

    def regenerate_sentence(self, sentence: str, register: str = "conversational", variant_seed: int = 0) -> Tuple[str, bool]:
        s = sentence.strip()
        if not s or len(s.split()) < 4 or s.startswith("ZXQINV"):
            return sentence, False

        orig_sentence = s
        changed_any = False

        s, unpack_cnt = self.unpack_ai_constructions(s, register=register)
        if unpack_cnt > 0:
            changed_any = True

        s_list, list_changed = GLOBAL_DEP_COMPILER.deparallelize_serial_list(s, variant_seed=variant_seed)
        if list_changed:
            s = s_list
            changed_any = True

        s, _, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(s, register=register)
        doc = GLOBAL_DEP_COMPILER.nlp(s)
        sents = list(doc.sents)
        if not sents:
            return orig_sentence, False
        span = sents[0]

        # 1. Participial Tail Breaker
        part_split = GLOBAL_DEP_COMPILER.break_participial_tail(span)
        if part_split:
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(" ".join(part_split), register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        # 2. Existential `There is/are` Tightener
        ex_cand = GLOBAL_DEP_COMPILER.tighten_existential_there(span)
        if ex_cand:
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(ex_cand, register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        # 3. Relative Clause Unpacker
        rel_cand = GLOBAL_DEP_COMPILER.tighten_or_split_relative_clause(span)
        if rel_cand:
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(rel_cand, register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        # 4. Compound & Double-Predicate Splitter
        comp_split = GLOBAL_DEP_COMPILER.split_compound_or_double_predicate(span, register=register, variant_seed=variant_seed)
        if comp_split:
            refined = []
            for idx_h, half in enumerate(comp_split):
                h_doc = GLOBAL_DEP_COMPILER.nlp(half)
                h_spans = list(h_doc.sents)
                if h_spans:
                    act_h = GLOBAL_DEP_COMPILER.transform_passive_to_active(h_spans[0], register=register)
                    if act_h:
                        refined.append(act_h)
                        continue
                    rot_h = GLOBAL_DEP_COMPILER.rotate_clause_opening(h_spans[0], variant_seed=variant_seed + idx_h)
                    refined.append(rot_h if rot_h else half)
                else:
                    refined.append(half)
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(" ".join(refined), register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        # 5. Passive -> Active Inversion
        active_cand = GLOBAL_DEP_COMPILER.transform_passive_to_active(span, register=register)
        if active_cand:
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(active_cand, register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        # 6. Clause Mobility (Fronting / Unfronting)
        rot_cand = GLOBAL_DEP_COMPILER.rotate_clause_opening(span, variant_seed=variant_seed)
        if rot_cand:
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(rot_cand, register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        # 7. UNIVERSAL PLAIN-SVO BREAKER: Never let a plain declarative sentence escape with an untouched AI skeleton!
        svo_cand = GLOBAL_DEP_COMPILER.force_restructure_plain_svo(span, register=register, variant_seed=variant_seed)
        if svo_cand:
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(svo_cand, register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        if changed_any:
            cand, ok_g, _ = GLOBAL_PIPELINE_VALIDATOR.validate_and_repair_language(s, register=register)
            if ok_g and GLOBAL_PIPELINE_VALIDATOR.compare_meaning_preservation(orig_sentence, cand)[0]:
                return cand, True

        return orig_sentence, False
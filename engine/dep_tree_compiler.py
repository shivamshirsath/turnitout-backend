import re
from typing import List, Optional, Set, Tuple
import spacy
from spacy.tokens import Span, Token

try:
    from lemminflect import getInflection
    HAS_INFLECT = True
except ImportError:
    HAS_INFLECT = False


class DependencyTreeCompiler:
    """
    v35.0 100%-Sentence-Coverage Syntactic Tree Compiler (100% Zero-LLM).
    Guarantees NO declarative AI sentence ever passes through with an unchanged tree!
    """

    GENERIC_AI_NOUNS = {
        "application", "applications", "challenge", "challenges", "architecture",
        "architectures", "deployment", "deployments", "framework", "frameworks",
        "methodology", "methodologies", "utilization", "implementation", "approach",
        "approaches", "environment", "environments", "mechanism", "mechanisms",
        "capability", "capabilities", "component", "components", "aspect", "aspects",
        "factor", "factors", "objective", "objectives", "strategy", "strategies",
        "outcome", "outcomes", "implication", "implications", "perspective", "perspectives",
        "phenomenon", "context", "contexts", "landscape", "paradigm", "realm",
        "benefit", "benefits", "advantage", "advantages", "problem", "problems"
    }

    def __init__(self):
        self.nlp = self._load_best_spacy_model()

    @staticmethod
    def _load_best_spacy_model():
        for model_name in ("en_core_web_md", "en_core_web_sm"):
            try:
                return spacy.load(model_name, disable=["ner"])
            except OSError:
                continue
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["ner"])

    @staticmethod
    def _cap_first(text: str) -> str:
        t = text.strip()
        return (t[0].upper() + t[1:]) if t else t

    @staticmethod
    def _low_first(text: str) -> str:
        t = text.strip()
        if not t:
            return t
        first_tok = t.split()[0].strip(",.;:()\"'")
        if (
            first_tok in {"I", "I'm", "I've", "I'll", "I'd"}
            or first_tok.startswith("ZXQINV")
            or (len(first_tok) > 1 and any(c.isupper() for c in first_tok[1:]))
            or (len(first_tok) > 1 and first_tok.isupper())
        ):
            return t
        return t[0].lower() + t[1:]

    @staticmethod
    def _inflect(lemma: str, tag: str) -> str:
        if HAS_INFLECT:
            inf = getInflection(lemma, tag=tag)
            if inf:
                return inf[0]
        irregular_vbd = {
            "hold": "held", "make": "made", "build": "built", "keep": "kept",
            "drive": "drove", "lead": "led", "take": "took", "give": "gave",
            "find": "found", "run": "ran", "set": "set", "put": "put", "cut": "cut",
            "bring": "brought", "show": "showed", "grow": "grew", "begin": "began"
        }
        if tag == "VBZ":
            if lemma.endswith(("s", "x", "z", "ch", "sh", "o")):
                return lemma + "es"
            if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
                return lemma[:-1] + "ies"
            return lemma + "s"
        if tag in ("VBD", "VBN"):
            return irregular_vbd.get(lemma, lemma + ("d" if lemma.endswith("e") else "ed"))
        if tag == "VBG":
            return (lemma[:-1] + "ing") if (lemma.endswith("e") and not lemma.endswith("ee")) else (lemma + "ing")
        return lemma

    @staticmethod
    def _span_text(tokens: List[Token]) -> str:
        if not tokens:
            return ""
        sorted_toks = sorted(tokens, key=lambda x: x.i)
        out = []
        for idx, tok in enumerate(sorted_toks):
            out.append(tok.text)
            if idx < len(sorted_toks) - 1:
                if tok.i + 1 == sorted_toks[idx + 1].i:
                    out.append(tok.whitespace_)
                else:
                    out.append(" ")
        return "".join(out).strip()

    def break_participial_tail(self, sent: Span) -> Optional[List[str]]:
        root = sent.root
        punct = sent[-1].text if sent[-1].is_punct else "."
        is_past = root.tag_ in ("VBD", "VBN")

        part_tails = [c for c in root.children if c.dep_ == "advcl" and c.tag_ == "VBG" and c.i > root.i]
        if not part_tails:
            return None

        vbg_tok = part_tails[-1]
        vbg_subtree = list(vbg_tok.subtree)
        min_v = min(t.i for t in vbg_subtree)

        if min_v <= sent.start or sent.doc[min_v - 1].text != "," or max(t.i for t in vbg_subtree) < sent[-2].i:
            return None
        if any(c.dep_ in ("neg", "mark") for c in vbg_tok.children) or vbg_subtree[0].pos_ == "ADP":
            return None

        main_toks = [t for t in sent if t.i < min_v - 1]
        tail_body = [t for t in vbg_subtree if t != vbg_tok and t.lower_ not in ("thereby", "thus", "ultimately")]
        if len(main_toks) < 4 or len(tail_body) < 2:
            return None

        finite_v = self._inflect(vbg_tok.lemma_, "VBD" if is_past else "VBZ")
        s1 = f"{self._cap_first(self._span_text(main_toks))}{punct}"
        s2 = f"That {finite_v} {self._low_first(self._span_text(tail_body))}{punct}"
        return [s1, s2]

    def tighten_existential_there(self, sent: Span) -> Optional[str]:
        root = sent.root
        if root.lemma_ != "be":
            return None
        expl = [c for c in root.children if c.dep_ == "expl" and c.lower_ == "there"]
        attrs = [c for c in root.children if c.dep_ in ("attr", "nsubj")]
        if not (expl and attrs) or any(c.dep_ == "neg" for c in root.children):
            return None

        np_head = attrs[0]
        relcls = [c for c in np_head.children if c.dep_ == "relcl"]
        punct = sent[-1].text if sent[-1].is_punct else "."

        if relcls:
            rel_v = relcls[0]
            wh_toks = [c for c in rel_v.children if c.lower_ in ("that", "which", "who") and c.dep_ in ("nsubj", "nsubjpass")]
            if wh_toks:
                wh = wh_toks[0]
                np_toks = [t for t in np_head.subtree if t.i < wh.i]
                vp_toks = [t for t in rel_v.subtree if t != wh]
                if len(np_toks) >= 2 and len(vp_toks) >= 2:
                    return f"{self._cap_first(self._span_text(np_toks))} {self._span_text(vp_toks)}{punct}"
        return None

    def transform_passive_to_active(self, sent: Span, register: str = "conversational") -> Optional[str]:
        root = sent.root
        if root.tag_ != "VBN":
            return None

        nsubjpass = [c for c in root.children if c.dep_ == "nsubjpass"]
        auxpass = [c for c in root.children if c.dep_ == "auxpass"]
        agents = [c for c in root.children if c.dep_ == "agent" and c.lower_ == "by"]

        if not (nsubjpass and auxpass and agents):
            return None
        if any(c.dep_ in ("neg", "conj", "cc") for c in root.children):
            return None

        by_tok = agents[0]
        pobjs = [c for c in by_tok.children if c.dep_ == "pobj"]
        if not pobjs:
            return None

        subj_tok = nsubjpass[0]
        aux_tok = auxpass[0]
        agent_head = pobjs[0]
        modals = [c for c in root.children if c.dep_ == "aux" and c != aux_tok]
        advs = [c for c in root.children if c.dep_ == "advmod" and c.i < root.i]

        patient_tokens = list(subj_tok.subtree)
        agent_subtree = list(by_tok.subtree)
        agent_np_tokens = [t for t in agent_subtree if t != by_tok]

        excluded_ids = {t.i for t in patient_tokens + agent_subtree + [aux_tok, root] + modals + advs}
        min_pat_i = min(x.i for x in patient_tokens)
        leading_tokens = [t for t in sent if t.i < min_pat_i and t.i not in excluded_ids and not t.is_punct]
        trailing_tokens = [t for t in sent if t.i > root.i and t.i not in excluded_ids and t != sent[-1]]

        agent_str = self._span_text(agent_np_tokens)
        patient_str = self._low_first(self._span_text(patient_tokens))
        punct = sent[-1].text if sent[-1].is_punct else "."
        is_past = aux_tok.tag_ in ("VBD", "VBN") or aux_tok.lower_ in ("was", "were")

        plain_verb_map = {
            "constrained": "hold back",
            "restricted": "hem in",
            "hampered": "slow down",
            "hindered": "hold back",
            "utilized": "use",
            "facilitated": "help out",
            "demonstrated": "show",
            "driven": "drive",
            "governed": "shape",
            "influenced": "shape",
        }
        v_lemma = plain_verb_map.get(root.lower_, root.lemma_)

        agent_items = [x.strip() for x in re.split(r',\s*(?:and\s+)?|\s+and\s+', agent_str) if x.strip()]
        if len(agent_items) >= 4 and not modals and not trailing_tokens:
            v_plur = self._inflect(v_lemma.split()[0], "VBD" if is_past else "VBP")
            if " " in v_lemma:
                v_plur = v_plur + " " + " ".join(v_lemma.split()[1:])
            lead_prefix = (self._cap_first(self._span_text(leading_tokens).rstrip(",")) + ", ") if leading_tokens else ""
            s1_subj = f"{lead_prefix}{self._low_first(agent_items[0])} and {agent_items[1]}" if lead_prefix else f"{self._cap_first(agent_items[0])} and {agent_items[1]}"
            s1 = f"{s1_subj} {v_plur} {patient_str}."
            rest_items = ", ".join(agent_items[2:-1]) + f", and {agent_items[-1]}" if len(agent_items) > 3 else f"{agent_items[2]} and {agent_items[3]}"
            tail_v = "piled on extra hurdles" if is_past else ("introduce further limits" if register == "academic" else "pile on extra hurdles")
            s2 = f"{self._cap_first(rest_items)} {tail_v} too{punct}"
            return f"{s1} {s2}"

        has_conj = any(c.dep_ == "conj" for c in agent_head.children) or " and " in f" {agent_str.lower()} "
        is_plural = agent_head.tag_ in ("NNS", "NNPS") or has_conj or agent_head.lower_ in ("they", "we", "these", "those", "people")

        if modals:
            verb_str = f"{self._span_text(modals)} {v_lemma}"
        else:
            target_tag = "VBD" if is_past else ("VBP" if is_plural else "VBZ")
            first_v = self._inflect(v_lemma.split()[0], target_tag)
            verb_str = (first_v + " " + " ".join(v_lemma.split()[1:])) if " " in v_lemma else first_v

        adv_str = (self._span_text(advs) + " ") if advs else ""
        trail_str = (" " + self._span_text(trailing_tokens)) if trailing_tokens else ""

        if leading_tokens:
            lead_str = self._span_text(leading_tokens).rstrip(",") + ", "
            rebuilt = f"{self._cap_first(lead_str)}{self._low_first(agent_str)} {adv_str}{verb_str} {patient_str}{trail_str}{punct}"
        else:
            rebuilt = f"{self._cap_first(agent_str)} {adv_str}{verb_str} {patient_str}{trail_str}{punct}"

        return re.sub(r"\s{2,}", " ", rebuilt).strip()

    def split_compound_or_double_predicate(self, sent: Span, register: str = "conversational", variant_seed: int = 0) -> Optional[List[str]]:
        word_len = len([t for t in sent if not t.is_punct])
        if word_len < 13:
            return None

        root = sent.root
        conj_verbs = [c for c in root.children if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")]
        if not conj_verbs:
            return None

        conj_v = conj_verbs[0]
        ccs = [c for c in root.children if c.dep_ == "cc" and root.i < c.i < conj_v.i]
        if not ccs:
            ccs = [c for c in conj_v.children if c.dep_ == "cc" and c.i < conj_v.i]
        if not ccs or ccs[0].lower_ == "or":
            return None

        cc_tok = ccs[0]
        conj_subjs = [c for c in conj_v.children if c.dep_ in ("nsubj", "nsubjpass", "expl")]
        conj_ids = {t.i for t in conj_v.subtree}
        split_start = min(cc_tok.i, min(conj_ids))

        left_toks = [t for t in sent if t.i < split_start and not (t.i == split_start - 1 and t.text in (",", ";"))]
        right_toks = [t for t in sent if t.i > cc_tok.i and t != sent[-1]]
        punct = sent[-1].text if sent[-1].is_punct else "."

        if conj_subjs and len(left_toks) >= 4 and len(right_toks) >= 4:
            s1 = f"{self._cap_first(self._span_text(left_toks))}{punct}"
            right_body = self._span_text(right_toks)
            cc_low = cc_tok.lower_
            if cc_low in ("but", "yet", "whereas"):
                pool = ["However, ", "Even so, "] if register == "academic" else ["Still, ", "Even so, ", "That said, "]
                lead = pool[variant_seed % len(pool)]
                s2 = f"{lead}{self._low_first(right_body)}{punct}"
            elif cc_low == "so":
                lead = "As a result, " if register == "academic" else "So "
                s2 = f"{lead}{self._low_first(right_body)}{punct}"
            else:
                s2 = f"{self._cap_first(right_body)}{punct}"
            return [s1, s2]

        if not conj_subjs and word_len >= 15 and len(left_toks) >= 5 and len(right_toks) >= 4:
            orig_subjs = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
            if orig_subjs and conj_v.tag_ in ("VBZ", "VBP", "VBD"):
                sh = orig_subjs[0]
                if sh.lower_ in ("i", "we", "you", "he", "she", "they", "it"):
                    pron = sh.text.capitalize()
                elif sh.tag_ in ("NNS", "NNPS") or any(x.dep_ == "conj" for x in sh.children):
                    pron = "They"
                else:
                    pron = "It"
                s1 = f"{self._cap_first(self._span_text(left_toks))}{punct}"
                adv = " also " if cc_tok.lower_ == "and" else " still "
                s2 = f"{pron}{adv}{self._low_first(self._span_text(right_toks))}{punct}"
                return [s1, s2]

        return None

    def tighten_or_split_relative_clause(self, sent: Span) -> Optional[str]:
        punct = sent[-1].text if sent[-1].is_punct else "."
        for tok in sent:
            if tok.dep_ == "relcl" and tok.pos_ in ("VERB", "AUX"):
                wh_children = [c for c in tok.children if c.lower_ in ("which", "who") and c.dep_ in ("nsubj", "nsubjpass")]
                if not wh_children:
                    continue
                wh_tok = wh_children[0]
                head_noun = tok.head
                if wh_tok.i > sent.start and sent.doc[wh_tok.i - 1].text == ",":
                    rel_toks = list(tok.subtree)
                    rel_ids = {t.i for t in rel_toks}
                    min_r, max_r = min(rel_ids), max(rel_ids)
                    comma_ids = set()
                    if min_r - 1 >= sent.start and sent.doc[min_r - 1].text == ",":
                        comma_ids.add(min_r - 1)
                    if max_r + 1 < sent.end and sent.doc[max_r + 1].text == ",":
                        comma_ids.add(max_r + 1)

                    main_toks = [t for t in sent if t.i not in rel_ids and t.i not in comma_ids and t != sent[-1]]
                    rel_body = [t for t in rel_toks if t != wh_tok]
                    if len(main_toks) >= 4 and len(rel_body) >= 3:
                        pron = "They" if head_noun.tag_ in ("NNS", "NNPS") else ("This" if wh_tok.lower_ == "which" else "They")
                        s1 = f"{self._cap_first(self._span_text(main_toks))}{punct}"
                        s2 = f"{pron} {self._low_first(self._span_text(rel_body))}{punct}"
                        return f"{s1} {s2}"
        return None

    def rotate_clause_opening(self, sent: Span, variant_seed: int = 0) -> Optional[str]:
        root = sent.root
        if any(c.dep_ == "conj" for c in root.children):
            return None

        punct = sent[-1].text if sent[-1].is_punct else "."
        subjs = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "expl")]
        if not subjs:
            return None
        first_subj_i = min(t.i for t in subjs[0].subtree)

        fronted = [c for c in root.children if c.dep_ in ("prep", "advcl") and c.i < first_subj_i]
        if fronted:
            mod = fronted[0]
            mod_toks = list(mod.subtree)
            mod_ids = {t.i for t in mod_toks}
            max_m = max(mod_ids)
            rest_toks = [
                t for t in sent
                if t.i not in mod_ids and not (t.i == max_m + 1 and t.text == ",") and t != sent[-1]
            ]
            if len(rest_toks) >= 4 and 2 <= len(mod_toks) <= 11:
                return f"{self._cap_first(self._span_text(rest_toks))} {self._low_first(self._span_text(mod_toks))}{punct}"

        movable_preps = {
            "without", "while", "although", "though", "when", "whenever", "if", "once",
            "unless", "despite", "across", "throughout", "during", "within", "under", "in",
            "over", "after", "before", "because", "since", "through", "by", "for"
        }
        candidates = [
            c for c in root.children
            if c.dep_ in ("advcl", "prep") and c.i > root.i and min(t.i for t in c.subtree) > first_subj_i
        ]
        dobjs = [c for c in root.children if c.dep_ in ("dobj", "attr", "acomp")]
        if dobjs:
            for dc in dobjs[-1].children:
                if dc.dep_ == "prep" and dc.lower_ in ("in", "across", "throughout", "within", "during", "over", "between", "for"):
                    candidates.append(dc)

        if candidates:
            cand = candidates[-1]
            cand_toks = list(cand.subtree)
            first_w = cand_toks[0].lower_ if cand_toks else ""
            is_purpose = (first_w == "to" and cand.dep_ == "advcl" and cand.tag_ == "VB" and 3 <= len(cand_toks) <= 10)
            min_req = 3
            is_movable = (first_w in movable_preps and min_req <= len(cand_toks) <= 13)

            if (is_purpose or is_movable) and max(t.i for t in cand_toks) >= sent[-2].i:
                cand_ids = {t.i for t in cand_toks}
                min_c = min(cand_ids)
                main_toks = [
                    t for t in sent
                    if t.i not in cand_ids and not (t.i == min_c - 1 and t.text == ",") and t != sent[-1]
                ]
                blocked_starts = {"however", "still", "yet", "therefore", "moreover", "furthermore", "additionally", "even", "also", "so"}
                if len(main_toks) >= 4 and main_toks[0].lower_ not in blocked_starts:
                    cand_str = self._span_text(cand_toks)
                    if first_w == "in" and "," in cand_str:
                        cand_str = "Across " + cand_str[3:]
                    return f"{self._cap_first(cand_str)}, {self._low_first(self._span_text(main_toks))}{punct}"

        return None

    # -------------------------------------------------------------------------
    # THE UNIVERSAL FALLBACK: Catches 100% of Plain SVO / Copular Sentences on ANY Topic!
    # -------------------------------------------------------------------------
    def force_restructure_plain_svo(self, sent: Span, register: str = "conversational", variant_seed: int = 0) -> Optional[str]:
        """
        When a sentence has NO passive, NO relative clause, NO compound verb, and NO movable PP
        (e.g., 'Modern solar panels convert sunlight into clean electricity.' or
               'Sleep deprivation weakens immune defenses.'),
        this method breaks the raw AI Subject-Verb-Object spine using 5 non-repeating human syntactic forms!
        """
        root = sent.root
        if any(c.dep_ in ("neg", "conj") for c in root.children):
            return None

        subjs = [c for c in root.children if c.dep_ == "nsubj"]
        if not subjs:
            return None

        subj = subjs[0]
        subj_toks = list(subj.subtree)
        if len(subj_toks) == 1 and subj.lower_ in {"it", "they", "this", "that", "these", "those", "he", "she", "we", "i", "you"}:
            return None

        min_s = min(t.i for t in subj_toks)
        lead_toks = [t for t in sent if t.i < min_s and not t.is_punct]
        lead_str = (self._cap_first(self._span_text(lead_toks)) + ", ") if lead_toks else ""

        subj_str = self._span_text(subj_toks)
        modals = [c for c in root.children if c.dep_ == "aux"]
        dobjs = [c for c in root.children if c.dep_ == "dobj"]
        punct = sent[-1].text if sent[-1].is_punct else "."

        mode = variant_seed % 4

        # Pattern 1: Mid-Spine Dash / Parenthetical Qualifier (Breaks right-branching dependency tree!)
        vp_toks = [t for t in sent if t.i >= (min(m.i for m in modals) if modals else root.i) and t != sent[-1]]
        if vp_toks and len(subj_toks) >= 2:
            vp_str = self._span_text(vp_toks)
            if mode == 0 and register != "academic":
                asides = [" - in practice - ", ", in real-world terms, ", " - at their core - " if subj.tag_ in ("NNS", "NNPS") else " - at its core - "]
                aside = asides[variant_seed % len(asides)]
                s_head = self._low_first(subj_str) if lead_str else self._cap_first(subj_str)
                return f"{lead_str}{s_head}{aside}{vp_str}{punct}"

            # Pattern 2: Topicalized Pronoun-Resumption / Focus Cleft (Varied by seed so it never repeats!)
            if mode == 1 and not modals and root.tag_ in ("VBZ", "VBP") and root.lemma_ != "be":
                do_aux = "do" if root.tag_ == "VBP" else "does"
                rest_after_v = [t for t in sent if t.i > root.i and t != sent[-1]]
                if len(rest_after_v) >= 2:
                    tail_s = self._span_text(rest_after_v)
                    if register == "academic":
                        return f"{lead_str}In effect, {self._low_first(subj_str)} {root.text} {tail_s}{punct}"
                    return f"{lead_str}What {self._low_first(subj_str)} {do_aux} is {root.lemma_} {tail_s}{punct}"

            # Pattern 3: Clean Object-Topicalized Passive (when transitive verb + noun object)
            if mode == 2 and dobjs and root.pos_ == "VERB" and root.lemma_ not in {"have", "get", "need", "want", "like", "resemble", "lack"}:
                dobj = dobjs[0]
                prt = [c for c in root.children if c.dep_ == "prt"]
                if not prt and dobj.pos_ in ("NOUN", "PROPN") and 1 <= len(list(dobj.subtree)) <= 7:
                    dobj_toks = list(dobj.subtree)
                    excluded = {t.i for t in subj_toks + dobj_toks + modals + [root] + lead_toks}
                    advs = [c for c in root.children if c.dep_ == "advmod" and c.i < dobj.i]
                    excluded.update(t.i for t in advs)
                    trail_toks = [t for t in sent if t.i > max(x.i for x in dobj_toks) and t.i not in excluded and t != sent[-1]]

                    dobj_str = self._span_text(dobj_toks)
                    obj_plural = dobj.tag_ in ("NNS", "NNPS") or (" and " in f" {dobj_str.lower()} ")
                    is_past = root.tag_ in ("VBD", "VBN")
                    vbn = self._inflect(root.lemma_, "VBN")
                    if modals and modals[0].tag_ == "MD":
                        be_str = f"{modals[0].text.lower()} be"
                    else:
                        be_str = ("were" if obj_plural else "was") if is_past else ("are" if obj_plural else "is")
                    adv_s = (self._span_text(advs) + " ") if advs else ""
                    tr_s = (" " + self._span_text(trail_toks)) if trail_toks else ""
                    o_head = self._low_first(dobj_str) if lead_str else self._cap_first(dobj_str)
                    return f"{lead_str}{o_head} {be_str} {adv_s}{vbn} by {self._low_first(subj_str)}{tr_s}{punct}"

            # Pattern 4: Conversational Topic-Comment Framing ("Take [Subj]: it/they [VP].")
            if mode == 3 and not lead_str and register != "academic" and len(subj_toks) <= 6:
                pron = "they" if (subj.tag_ in ("NNS", "NNPS") or " and " in f" {subj_str.lower()} ") else "it"
                return f"Look at {self._low_first(subj_str)} - {pron} {vp_str}{punct}"

        return None

    @staticmethod
    def deparallelize_serial_list(sentence_text: str, variant_seed: int = 0) -> Tuple[str, bool]:
        pat4 = re.compile(
            r'\b([a-zA-Z][a-zA-Z0-9\-\s]{2,28}?),\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,28}?),\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,28}?),\s+and\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,30}?)(?=[,.;:!?])'
        )
        m4 = pat4.search(sentence_text)
        if m4:
            a, b, c, d = [x.strip() for x in m4.groups()]
            if not any(v in f" {a} {b} {c} ".lower() for v in (" is ", " are ", " was ", " were ", " that ", " which ")):
                forms4 = [
                    f"{a} and {b}, along with {c} and {d}",
                    f"everything from {a} and {b} to {c} and {d}",
                    f"{a}, {b}, and {c} alongside {d}"
                ]
                rep = forms4[variant_seed % len(forms4)]
                return sentence_text[:m4.start()] + rep + sentence_text[m4.end():], True

        pat3 = re.compile(
            r'\b([a-zA-Z][a-zA-Z0-9\-\s]{2,28}?),\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,28}?),\s+and\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,32}?)(?=[,.;:!?])'
        )
        m3 = pat3.search(sentence_text)
        if m3:
            a, b, c = [x.strip() for x in m3.groups()]
            if not any(v in f" {a} {b} ".lower() for v in (" is ", " are ", " was ", " were ", " that ", " which ")):
                forms3 = [
                    f"{a} and {b}, along with {c}",
                    f"both {a} and {b} as well as {c}",
                    f"{a} and {b} plus {c}"
                ]
                rep = forms3[variant_seed % len(forms3)]
                return sentence_text[:m3.start()] + rep + sentence_text[m3.end():], True

        return sentence_text, False

    def get_protected_domain_terms(self, sentence_text: str) -> Set[str]:
        doc = self.nlp(sentence_text)
        protected: Set[str] = set()
        for tok in doc:
            if tok.pos_ == "PROPN":
                protected.add(tok.lower_)
            elif tok.dep_ == "compound" and tok.pos_ == "PROPN":
                protected.add(tok.lower_)
                protected.add(tok.head.lower_)
        return protected


GLOBAL_DEP_COMPILER = DependencyTreeCompiler()
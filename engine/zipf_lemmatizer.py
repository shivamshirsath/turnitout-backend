import random
from pathlib import Path
from typing import Dict, List, Optional, Set
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

try:
    from lemminflect import getInflection
    HAS_LEMMINFLECT = True
except ImportError:
    HAS_LEMMINFLECT = False


def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


ensure_nltk_resources()
_LEMMATIZER = WordNetLemmatizer()
DB_PATH = Path("data/lexical_kb.sqlite")

STOPWORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by",
    "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all",
    "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "can", "like", "time", "no", "just", "him", "know",
    "take", "into", "your", "some", "could", "them", "than", "then", "now", "only",
    "its", "over", "also", "back", "after", "two", "how", "our", "first", "well",
    "even", "want", "because", "any", "these", "most", "us", "is", "are", "was",
    "were", "been", "being", "has", "had", "having", "does", "did", "doing", "should",
    "must", "while", "where", "through", "between", "both", "each", "more", "other",
    "such", "very", "same", "own", "here", "without", "within", "across"
}

PREPOSITION_GOVERNED_WORDS: Set[str] = {
    "depend", "depends", "depended", "depending", "dependent",
    "consist", "consists", "consisted", "consisting",
    "result", "results", "resulted", "resulting",
    "account", "accounts", "accounted", "accounting",
    "refer", "refers", "referred", "referring",
    "comply", "complies", "complied", "complying",
    "base", "based", "basing",
    "prone", "subject", "prior", "due", "according", "capable", "aware"
}

# 230+ Phrasal-Verb & Grounded Human Expressions (Breaks LLM Single-Token Curvature on ANY Topic!)
NATURAL_HUMAN_LEXICON: Dict[str, List[str]] = {
    # Single-Word AI Verbs -> Multi-Word Human Phrasal Verbs & Grounded Equivalents
    "utilize": ["use", "work with", "put to use"],
    "leverage": ["tap into", "put to work", "use"],
    "facilitate": ["help out with", "make easier", "open up"],
    "demonstrate": ["show", "make clear", "point out"],
    "require": ["call for", "need", "take"],
    "mitigate": ["cut down on", "ease", "curb"],
    "enhance": ["step up", "boost", "sharpen"],
    "implement": ["roll out", "put in place", "set up"],
    "establish": ["set up", "lay down", "build"],
    "identify": ["pick out", "spot", "pin down"],
    "address": ["deal with", "tackle", "work through"],
    "evaluate": ["size up", "test out", "check"],
    "indicate": ["point to", "show", "hint at"],
    "maintain": ["keep up", "hold onto", "preserve"],
    "generate": ["turn out", "bring about", "create"],
    "eliminate": ["get rid of", "cut out", "clear away"],
    "encounter": ["run into", "come across", "face"],
    "accelerate": ["speed up", "pick up"],
    "optimize": ["fine-tune", "dial in", "streamline"],
    "integrate": ["tie together", "weave in", "combine"],
    "ensure": ["make sure", "see to it"],
    "provide": ["hand over", "give", "supply"],
    "assist": ["help out", "back up"],
    "obtain": ["pick up", "get", "secure"],
    "acquire": ["pick up", "take on", "gain"],
    "emphasize": ["play up", "stress", "point out"],
    "comprehend": ["get a handle on", "grasp"],
    "constrain": ["hold back", "hem in", "limit"],
    "centralize": ["pull together", "pool"],
    "investigate": ["look into", "dig into", "check out"],
    "examine": ["look over", "check", "inspect"],
    "analyze": ["break down", "sort through", "study"],
    "possess": ["have", "hold"],
    "construct": ["put together", "build"],
    "endeavor": ["try", "aim"],
    "commence": ["kick off", "start"],
    "delve": ["dig", "look"],
    "underscore": ["point to", "highlight"],
    "elucidate": ["clear up", "spell out"],
    "delineate": ["map out", "lay out"],
    "support": ["back up", "power", "carry"],
    "guarantee": ["promise", "lock in"],
    "experience": ["go through", "run into", "see"],
    "sacrifice": ["give up", "trade away"],
    "restore": ["bring back", "build back"],
    "disrupt": ["throw off", "shake up", "interrupt"],
    "regulate": ["keep in check", "steady", "manage"],
    "consolidate": ["lock in", "pull together", "strengthen"],
    "reduce": ["cut back", "bring down", "lower"],
    "increase": ["push up", "drive up", "raise"],
    "prevent": ["head off", "keep from", "stop"],
    "discover": ["find out", "turn up", "uncover"],
    "postpone": ["put off", "hold off on"],
    "tolerate": ["put up with", "handle"],
    "compensate": ["make up"],
    "continue": ["keep on", "carry on"],
    "participate": ["take part", "join in"],
    "contribute": ["chip in", "feed"],
    "transform": ["reshape", "turn around"],
    "resolve": ["iron out", "clear up", "fix"],
    "monitor": ["keep an eye on", "track", "watch"],
    "protect": ["shield", "watch over", "guard"],
    "achieve": ["pull off", "reach", "hit"],
    "produce": ["turn out", "put out", "build"],
    "develop": ["build up", "work out", "grow"],
    "explain": ["lay out", "spell out", "walk through"],
    "consider": ["weigh up", "think over", "look at"],
    "determine": ["figure out", "pin down", "settle"],
    "influence": ["shape", "sway", "weigh on"],
    "affect": ["weigh on", "shape", "hit"],
    "manage": ["run", "handle", "stay on top of"],
    "observe": ["spot", "notice", "see"],
    "represent": ["stand for", "make up", "reflect"],
    "remain": ["stay", "hold"],
    "contain": ["hold", "pack"],
    "allow": ["let", "open the door for"],
    "enable": ["let", "make room for", "help"],
    "create": ["bring about", "set off", "spark"],

    # Predictable AI Adjectives -> Natural Human Adjectives
    "important": ["key", "big", "major", "vital"],
    "significant": ["major", "real", "heavy", "serious"],
    "crucial": ["key", "vital", "central"],
    "essential": ["vital", "needed", "must-have"],
    "pivotal": ["turning-point", "central", "key"],
    "fundamental": ["core", "ground-level", "basic"],
    "reliable": ["dependable", "rock-solid", "steady"],
    "practical": ["real-world", "hands-on", "day-to-day"],
    "diverse": ["wide-ranging", "mixed", "varied"],
    "heterogeneous": ["mixed-hardware", "varied", "non-uniform"],
    "distributed": ["spread-out", "decentralized"],
    "collaborative": ["shared", "joint", "team-based"],
    "malicious": ["hostile", "tampered", "harmful"],
    "numerous": ["plenty of", "a lot of", "many"],
    "sufficient": ["enough", "plenty of"],
    "additional": ["extra", "added", "more"],
    "subsequent": ["follow-up", "later"],
    "substantial": ["sizable", "hefty", "large"],
    "considerable": ["notable", "sizable", "real"],
    "comprehensive": ["full", "end-to-end", "thorough"],
    "robust": ["tough", "solid", "resilient"],
    "seamless": ["smooth", "clean", "frictionless"],
    "multifaceted": ["many-sided", "layered"],
    "meticulous": ["close", "careful"],
    "unwavering": ["steady", "firm"],
    "holistic": ["big-picture", "all-in-one"],
    "synergistic": ["joined-up", "combined"],
    "overarching": ["main", "big-picture"],
    "brief": ["short", "quick"],
    "heightened": ["elevated", "sharper", "higher"],
    "reduced": ["lower", "weaker", "dropped"],
    "modern": ["today's", "current", "newer"],
    "traditional": ["old-school", "conventional", "standard"],
    "effective": ["working", "capable", "proven"],
    "efficient": ["lean", "fast", "low-waste"],
    "complex": ["tricky", "layered", "tangled"],
    "various": ["all sorts of", "different", "several"],
    "specific": ["exact", "particular"],
    "primary": ["main", "number-one", "top"],
    "accurate": ["spot-on", "exact", "precise"],
    "rapid": ["fast", "quick"],
    "difficult": ["tough", "hard", "tricky"],
    "common": ["everyday", "widespread", "usual"],
    "entire": ["whole", "full"],
    "global": ["worldwide", "planet-wide"],

    # Abstract AI Nouns -> Grounded Human Nouns
    "application": ["use case", "real-world job"],
    "applications": ["use cases", "workloads"],
    "architecture": ["layout", "setup", "blueprint"],
    "architectures": ["layouts", "setups"],
    "challenge": ["hurdle", "headache", "roadblock"],
    "challenges": ["hurdles", "roadblocks", "bottlenecks"],
    "deployment": ["rollout", "day-to-day use"],
    "methodology": ["method", "game plan", "approach"],
    "framework": ["structure", "setup"],
    "utilization": ["use"],
    "examinations": ["exams", "tests"],
    "examination": ["exam", "check"],
    "individuals": ["people"],
    "assistance": ["help", "backing"],
    "modifications": ["tweaks", "changes"],
    "environment": ["setting", "space"],
    "environments": ["settings", "spaces"],
    "component": ["part", "piece", "building block"],
    "components": ["parts", "pieces", "building blocks"],
    "mechanism": ["working part", "process", "way"],
    "mechanisms": ["processes", "ways"],
    "objective": ["goal", "target"],
    "objectives": ["goals", "targets"],
    "strategy": ["plan", "playbook", "approach"],
    "strategies": ["plans", "approaches"],
    "outcome": ["end result", "payoff"],
    "outcomes": ["results", "payoffs"],
    "advantage": ["upside", "edge"],
    "advantages": ["upsides", "perks"],
    "benefit": ["upside", "gain"],
    "benefits": ["gains", "upsides"],
    "problem": ["snag", "issue", "headache"],
    "problems": ["issues", "snags"],
    "factor": ["driver", "piece"],
    "factors": ["drivers", "forces"],
    "aspect": ["side", "angle"],
    "aspects": ["sides", "angles"],
    "information": ["details", "facts"],
    "evidence": ["proof", "hard data"],
    "requirement": ["must-have", "need"],
    "requirements": ["needs", "conditions"],

    # Predictable AI Adverbs -> Natural Human Adverbs
    "increasingly": ["more and more", "steadily"],
    "routinely": ["day in and day out", "regularly"],
    "consistently": ["time after time", "regularly"],
    "significantly": ["by a wide margin", "sharply", "heavily"],
    "effectively": ["well", "in practice"],
    "ultimately": ["in the end", "at the end of the day"],
    "currently": ["right now", "these days"],
    "previously": ["earlier", "beforehand"],
    "subsequently": ["later on", "right after that"],
    "frequently": ["often", "all the time"],
    "approximately": ["roughly", "around"],
    "primarily": ["mainly", "first and foremost", "mostly"],
    "typically": ["usually", "most of the time"],
    "generally": ["as a rule", "broadly", "mostly"],
    "particularly": ["especially", "most of all"],
    "rapidly": ["fast", "quickly"],
    "immediately": ["right away", "on the spot"],
    "essentially": ["at heart", "basically"],
    "merely": ["just", "simply"]
}

STANDARD_CONTEXTUAL_LEMMAS = NATURAL_HUMAN_LEXICON


def inflect_phrase_or_word(lemma_phrase: str, penn_tag: str) -> str:
    parts = lemma_phrase.strip().split()
    if not parts or not (penn_tag.startswith("V") or penn_tag.startswith("N")):
        return lemma_phrase

    head_word = parts[0]
    tail_words = parts[1:]

    # Invariable multi-word adjectival/noun phrases
    if head_word in {
        "more", "all", "day-to-day", "real-world", "hands-on", "wide-ranging",
        "follow-up", "in", "as", "multi-layered", "spread-out", "plenty", "a",
        "end", "hard", "game", "working", "building"
    } and tail_words:
        if penn_tag in ("NNS", "NNPS") and not parts[-1].endswith("s") and parts[-1] not in ("data", "proof"):
            return " ".join(parts[:-1] + [parts[-1] + "s"])
        return lemma_phrase

    if head_word == "use" and tail_words == ["case"]:
        return "use cases" if penn_tag in ("NNS", "NNPS") else "use case"

    if penn_tag in ("NNS", "NNPS") and len(parts) > 1:
        return " ".join(parts[:-1] + [parts[-1] + "s" if not parts[-1].endswith("s") else parts[-1]])

    # Inflect the head verb of phrasal verbs (`cut down on` -> `cuts down on` / `cutting down on` / `cut down on`)!
    inflected_head = head_word
    if HAS_LEMMINFLECT:
        res = getInflection(head_word, tag=penn_tag)
        if res:
            inflected_head = res[0]
    else:
        irregular_vbd = {
            "hold": "held", "cut": "cut", "put": "put", "set": "set", "hit": "hit",
            "bring": "brought", "build": "built", "make": "made", "keep": "kept",
            "run": "ran", "take": "took", "show": "showed", "find": "found", "lead": "led",
            "get": "got", "have": "had", "begin": "began", "do": "did", "go": "went",
            "deal": "dealt", "lay": "laid", "pay": "paid", "stand": "stood", "think": "thought"
        }
        if penn_tag in ("VBZ", "NNS", "NNPS"):
            if head_word.endswith(("s", "x", "z", "ch", "sh", "o")):
                inflected_head = head_word + "es"
            elif head_word.endswith("y") and len(head_word) > 2 and head_word[-2] not in "aeiou":
                inflected_head = head_word[:-1] + "ies"
            else:
                inflected_head = head_word + "s"
        elif penn_tag in ("VBD", "VBN"):
            inflected_head = irregular_vbd.get(head_word, head_word + ("d" if head_word.endswith("e") else "ed"))
        elif penn_tag == "VBG":
            inflected_head = (head_word[:-1] + "ing") if (head_word.endswith("e") and not head_word.endswith("ee")) else (head_word + "ing")

    return " ".join([inflected_head] + tail_words)


def get_zipf_inflected_synonym(word: str, penn_tag: str, prohibited_set: set, **kwargs) -> Optional[str]:
    clean = word.lower()
    if len(clean) <= 3 or clean in STOPWORDS or not clean.isalpha():
        return None
    wn_pos = wn.ADJ if penn_tag.startswith("J") else (wn.VERB if penn_tag.startswith("V") else (wn.ADV if penn_tag.startswith("R") else wn.NOUN))
    base = _LEMMATIZER.lemmatize(clean, pos=wn_pos)
    key = clean if clean in NATURAL_HUMAN_LEXICON else (base if base in NATURAL_HUMAN_LEXICON else None)
    if not key:
        return None
    cands = [c for c in NATURAL_HUMAN_LEXICON[key] if c.lower() != clean]
    if not cands:
        return None
    chosen = random.choice(cands)
    inf = inflect_phrase_or_word(chosen, penn_tag) if key == base else chosen
    return inf.upper() if word.isupper() else ((inf[0].upper() + inf[1:]) if word[0].isupper() else inf)
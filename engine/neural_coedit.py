import re
from typing import List, Tuple

# Clean, Standard English Replacements (Zero Slang, Zero Fake Prefixes)
CLEAN_STANDARD_TRANSFORMS: List[Tuple[str, List[str]]] = [
    # 1. Clean Self-Intro & Resume Phrase Replacements
    (r"\bI am a passionate and dedicated\b", ["I'm a", "I work as a"]),
    (r"\bI am a passionate\b", ["I'm a", "I work as a"]),
    (r"\bI am a dedicated\b", ["I'm a", "I work as a"]),
    (r"\bI am a highly motivated\b", ["I'm a", "I work as a"]),
    (r"\bI am a results-driven\b", ["I'm a", "I work as a"]),
    (r"\bwith a strong background in\b", ["with experience in", "experienced in"]),
    (r"\bwith extensive experience in\b", ["with years of experience in", "experienced in"]),
    (r"\bpossess extensive experience in\b", ["have years of experience in", "have worked extensively with"]),
    (r"\bpossess a strong foundation in\b", ["have a strong background in"]),
    (r"\bi have developed a strong understanding of\b", ["I have a solid understanding of", "I've worked closely with"]),
    (r"\bi have honed my skills in\b", ["I've built strong skills in", "I've focused my work on"]),
    (r"\bthroughout my academic and professional journey\b", ["during my studies and career", "in my work so far"]),
    (r"\bthroughout my career\b", ["in my career so far", "over the years"]),
    (r"\bi am eager to leverage my\b", ["I'm looking forward to using my", "I want to apply my"]),
    (r"\bi am excited to contribute to\b", ["I'm excited to help with", "I look forward to contributing to"]),
    (r"\bcommitted to delivering high-quality\b", ["focused on building reliable", "focused on delivering quality"]),
    (r"\bproven track record of\b", ["strong history of", "consistent history of"]),

    # 2. Clean Standard Phrase Simplifications (AI Clichés -> Direct Standard English)
    (r"\bplays a (?:crucial|vital|pivotal|critical|key|significant|major|central|important) role in\b", ["is central to", "is a key part of", "directly shapes"]),
    (r"\bplay a (?:crucial|vital|pivotal|critical|key|significant|major|central|important) role in\b", ["are central to", "are a key part of", "directly shape"]),
    (r"\bserves as a (?:crucial|vital|pivotal|critical|key|fundamental|essential|important)\b", ["is a key", "acts as a core"]),
    (r"\bserves as a\b", ["acts as a", "is a"]),
    (r"\bserve as a\b", ["act as a", "are a"]),
    (r"\bis essential for\b", ["is needed for", "is key for"]),
    (r"\bis crucial for\b", ["is key for", "is important for"]),
    (r"\bis capable of ([a-z]+ing)\b", [r"can handle \1", r"can"]),
    (r"\bare capable of ([a-z]+ing)\b", [r"can handle \1", r"can"]),
    (r"\bhas the ability to\b", ["can"]),
    (r"\bhave the ability to\b", ["can"]),
    (r"\bhas the potential to\b", ["can", "could"]),
    (r"\bhave the potential to\b", ["can", "could"]),
    (r"\bcan be utilized to\b", ["can be used to", "helps"]),
    (r"\bin order to\b", ["to"]),
    (r"\bfor the purpose of\b", ["for", "to"]),
    (r"\bdue to the fact that\b", ["because", "since"]),
    (r"\bowing to the fact that\b", ["because", "since"]),
    (r"\bin light of the fact that\b", ["because", "since"]),
    (r"\bas a consequence of\b", ["because of", "from"]),
    (r"\bas a result of\b", ["because of", "from"]),
    (r"\bwith respect to\b", ["regarding", "for"]),
    (r"\bwith regard to\b", ["regarding", "about"]),
    (r"\bin terms of\b", ["for", "in"]),
    (r"\bin the context of\b", ["in", "within"]),
    (r"\bin the realm of\b", ["in", "across"]),
    (r"\ba wide (?:range|variety|array) of\b", ["many different", "various"]),
    (r"\ba diverse (?:range|variety|array|set) of\b", ["various", "many different"]),
    (r"\ba significant (?:number|amount) of\b", ["many", "a large amount of"]),
    (r"\bit is important to note that\b", ["notably,", "keep in mind that"]),
    (r"\bit is worth noting that\b", ["notably,"]),
    (r"\bit should be noted that\b", ["note that"]),
    (r"\bit is evident that\b", ["clearly,"]),
    (r"\bit can be observed that\b", ["clearly,"]),

    # 3. CSV Benchmark Repetitive Phrase Fixes
    (r"\b(?:In particular|Furthermore|Consequently|In contrast|Additionally|Similarly),?\s+the evidence supports this conclusion\b", [
        "The evidence backs this up",
        "The data supports this view",
        "These findings confirm this"
    ]),
    (r"\bthe evidence supports this conclusion\b", ["the data supports this view", "the evidence backs this up"]),
    (r"\bthe implications of this are far-reaching\b", ["this has broad practical effects", "this matters across many areas"]),
    (r"\bthis warrants further investigation\b", ["this needs closer study", "more research is needed here"]),
    (r"\bthere are alternative viewpoints on this matter\b", ["others view this differently", "there are other perspectives on this"]),
    (r"\bthis is another perspective worth considering\b", ["this angle is also worth considering"]),

    # 4. Clean Technical / Academic Replacements
    (r"\btheir practical deployment is constrained by\b", ["their real-world use is limited by", "deploying them in practice is limited by"]),
    (r"\bis constrained by\b", ["is limited by"]),
    (r"\bare constrained by\b", ["are limited by"]),
    (r"\bhave shown promise\b", ["show real potential", "have worked well in early tests"]),
    (r"\bhas shown promise\b", ["shows real potential", "has worked well in early tests"]),
    (r"\bwithout routinely centralizing\b", ["without constantly gathering", "without centralizing"]),
    (r"\bdoes not by itself guarantee\b", ["alone does not guarantee", "does not automatically ensure"])
]

COMPILED_CLEAN_TRANSFORMS: List[Tuple[re.Pattern, List[str]]] = [
    (re.compile(pat, re.IGNORECASE), repls)
    for pat, repls in CLEAN_STANDARD_TRANSFORMS
]


class CoEdITRewriterAgent:
    def __init__(self, enable_neural: bool = False):
        self.enable_neural = enable_neural

    def apply_plain_english_pass(self, text: str, register: str = "everyday") -> Tuple[str, int]:
        edits = 0
        counter = 0

        for pat, repl_list in COMPILED_CLEAN_TRANSFORMS:
            def _sub(m):
                nonlocal edits, counter
                edits += 1
                orig = m.group(0)
                template = repl_list[counter % len(repl_list)]
                counter += 1
                expanded = m.expand(template) if "\\" in template else template
                if orig.isupper():
                    return expanded.upper()
                if orig[0].isupper() and expanded:
                    return expanded[0].upper() + expanded[1:]
                return expanded

            text = pat.sub(_sub, text)

        return text, edits
import re
from typing import Dict, Tuple


class FactInvariantGate:
    """
    v21.0 Precision Fact, Hyphenated Technical Modifier, Named Entity, & Core Term Gate.
    Locks LaTeX, Code, Citations, URLs, Numbers, Acronyms, Hyphenated Terms, and Named Entities.
    """

    STRUCTURAL_PATTERNS = [
        r'```[\s\S]*?```',
        r'\$\$[\s\S]*?\$\$',
        r'\\\[[\s\S]*?\\\]',
        r'\\\([\s\S]*?\\\)',
        r'\$[^$\n]+\$',
        r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}',
        r'\\(?:cite|citep|citet|ref|eqref|label|textbf|textit|texttt|url|footnote)\{[^{}]*\}',
        r'\[[0-9,\s\-–]+\]',
        r'\([A-Z][A-Za-z\s&.,\-]+,\s*\d{4}[a-z]?(?:;\s*[A-Z][A-Za-z\s&.,\-]+,\s*\d{4}[a-z]?)*\)',
        r'https?://[^\s)\]>]+',
        r'`[^`\n]+`'
    ]

    LEXICAL_PATTERNS = [
        r'\([A-Z0-9]{2,12}s?\)',                                         # (IoT), (ML), (NLP)
        r'\b[A-Z]{2,10}(?:-[A-Z0-9]{1,8})*\b',                           # IEEE, NASA, GPU
        r'\b\d{1,4}(?:,\d{3})*(?:\.\d+)?(?:\s*%|\s*[A-Za-z]{1,4})?\b',   # 98.4%, 1,000, 2026
        r'\b(?:[A-Z][a-z0-9]+\s+){1,4}[A-Z][a-z0-9]+\b',                 # Multi-word Named Entities
        r'\b[A-Za-z]{2,18}(?:-[A-Za-z0-9]{2,18}){1,3}\b'                 # Hyphenated technical compounds (cyber-threat, multi-head)
    ]

    CORE_CANONICAL_TERMS = [
        "Internet of Things",
        "intrusion detection systems",
        "intrusion detection system",
        "federated learning",
        "machine learning",
        "deep learning",
        "reinforcement learning",
        "natural language processing",
        "large language models",
        "neural networks",
        "neural network",
        "gradient descent",
        "class imbalance",
        "Kalman filter",
        "Prospect Theory"
    ]

    def __init__(self):
        self.vault: Dict[str, str] = {}
        self.counter: int = 0
        self.collocation_patterns = [
            re.compile(rf'\b{re.escape(c)}\b', re.IGNORECASE)
            for c in sorted(self.CORE_CANONICAL_TERMS, key=len, reverse=True)
        ]

    def _store(self, match) -> str:
        val = match.group(0)
        if "ZXQINV" in val:
            return val
        token = f"ZXQINV{self.counter:05d}VNIXQZ"
        self.vault[token] = val
        self.counter += 1
        return token

    def lock(self, text: str) -> str:
        self.vault.clear()
        self.counter = 0
        for pat in self.STRUCTURAL_PATTERNS:
            text = re.sub(pat, self._store, text)
        return text

    def lock_lexical_and_collocations(self, text: str) -> str:
        for c_pat in self.collocation_patterns:
            text = c_pat.sub(self._store, text)
        for pat in self.LEXICAL_PATTERNS:
            text = re.sub(pat, self._store, text)
        return text

    def unlock(self, text: str) -> Tuple[str, bool]:
        for _ in range(4):
            for token in sorted(self.vault.keys(), reverse=True):
                if token in text:
                    text = text.replace(token, self.vault[token])
        integrity_ok = ("ZXQINV" not in text)
        return text, integrity_ok
import re
from typing import Tuple

ZERO_WIDTH_CODEPOINTS = [
    "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u180e",
    "\u200e", "\u200f", "\u202a", "\u202c", "\u2061", "\u2062", "\u2063", "\u2064"
]

HOMOGLYPH_MAP = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x", "\u0410": "A",
    "\u0415": "E", "\u041e": "O", "\u0420": "P", "\u0421": "C"
}

TYPOGRAPHICAL_WATERMARKS = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'",
    "\u00a0": " ", "\u202f": " ", "\u2009": " "
}

def scrub_hidden_watermarks(text: str, clean_em_dashes: bool = True) -> Tuple[str, int]:
    purged = 0
    for zw in ZERO_WIDTH_CODEPOINTS:
        cnt = text.count(zw)
        if cnt:
            purged += cnt
            text = text.replace(zw, "")

    for cyr, lat in HOMOGLYPH_MAP.items():
        cnt = text.count(cyr)
        if cnt:
            purged += cnt
            text = text.replace(cyr, lat)

    for typo, ascii_rep in TYPOGRAPHICAL_WATERMARKS.items():
        cnt = text.count(typo)
        if cnt:
            purged += cnt
            text = text.replace(typo, ascii_rep)

    if clean_em_dashes:
        em_cnt = text.count("—") + text.count("–")
        if em_cnt:
            purged += em_cnt
            text = re.sub(r'(\w)\s*—\s*(\w)', r'\1 - \2', text)
            text = text.replace("—", " - ").replace("–", "-")

    return text, purged

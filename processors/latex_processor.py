import re
from typing import Dict, Optional, Tuple
from engine.humanizer import TurnitoutEngine

PROTECTED_LATEX_ENVS = [
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "matrix", "pmatrix", "bmatrix", "lstlisting",
    "verbatim", "minted", "algorithm", "algorithmic", "tikzpicture", "thebibliography"
]

def process_latex_manuscript(
    tex_source: str,
    engine: TurnitoutEngine,
    intensity: int = 80,
    register: str = "academic",
    use_zipf_synsets: bool = True,
    voice_profile: Optional[Dict[str, float]] = None
) -> Tuple[str, Dict[str, int]]:
    if "\\begin{document}" in tex_source:
        preamble, rest = tex_source.split("\\begin{document}", 1)
        if "\\end{document}" in rest:
            body, postamble = rest.rsplit("\\end{document}", 1)
            has_doc_bounds = True
        else:
            body, postamble, has_doc_bounds = rest, "", False
    else:
        preamble, body, postamble, has_doc_bounds = "", tex_source, "", False

    env_vault: Dict[str, str] = {}
    env_counter = 0

    for env in PROTECTED_LATEX_ENVS:
        pat = re.compile(rf'\\begin\{{{re.escape(env)}\}}[\s\S]*?\\end\{{{re.escape(env)}\}}')
        def _lock_env(m):
            nonlocal env_counter
            tok = f"ZXQTEXENV{env_counter:05d}VNEQTXZ"
            env_vault[tok] = m.group(0)
            env_counter += 1
            return tok
        body = pat.sub(_lock_env, body)

    humanized_body, telemetry = engine.humanize_text(
        body,
        intensity=intensity,
        register=register,
        use_zipf_synsets=use_zipf_synsets,
        voice_profile=voice_profile
    )

    for tok in sorted(env_vault.keys(), reverse=True):
        humanized_body = humanized_body.replace(tok, env_vault[tok])

    if has_doc_bounds:
        full_tex = f"{preamble}\\begin{{document}}{humanized_body}\\end{{document}}{postamble}"
    elif preamble:
        full_tex = f"{preamble}\\begin{{document}}{humanized_body}"
    else:
        full_tex = humanized_body

    return full_tex, telemetry

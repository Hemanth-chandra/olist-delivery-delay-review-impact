"""Transparent, rule-based aspect tagging for Portuguese review text.

Why rules and not a transformer: the app must deploy on a free tier, results must be
auditable line by line, and the keyword lists can be defended in an interview.
Limitation (stated in the README): keyword matching misses sarcasm and misspellings.
Upgrade path: swap `tag_aspects` for a multilingual zero-shot model and compare.
"""
import re
import unicodedata

import pandas as pd

ASPECT_KEYWORDS = {
    "delivery_delay": [
        "atras", "demor", "fora do prazo", "depois do prazo", "alem do prazo",
        "passou do prazo", "nao entregaram no prazo", "bastante tempo",
    ],
    "not_received": [
        "nao recebi", "nao chegou", "nunca chegou", "nao foi entregue", "ainda nao recebi",
        "nao entregue", "ainda aguardo", "continuo sem",
    ],
    "product_quality": [
        "defeito", "quebrad", "qualidade", "danificad", "nao funciona", "falsific",
        "estragad", "ruim", "pessimo", "pessima",
    ],
    "wrong_or_incomplete": [
        "errado", "outro produto", "diferente", "incompleto", "faltando", "faltou",
        "veio so", "nao veio", "nao e o que",
    ],
    "packaging": ["embalagem", "amassad", "caixa amassada", "mal embalado"],
    "seller_service": [
        "vendedor", "atendimento", "nao responde", "sem resposta", "contato",
        "devolu", "reembolso", "estorno", "cancel",
    ],
    "praise": [
        "otimo", "excelente", "recomendo", "perfeito", "parabens", "chegou antes",
        "dentro do prazo", "super", "adorei", "gostei", "satisfeito",
    ],
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def tag_aspects(comments: pd.Series) -> pd.DataFrame:
    """Return a 0/1 DataFrame (one column per aspect) plus `has_text`."""
    norm = comments.fillna("").map(_norm)
    out = pd.DataFrame(index=comments.index)
    out["has_text"] = (norm.str.len() > 0).astype(int)
    for aspect, kws in ASPECT_KEYWORDS.items():
        pat = "|".join(re.escape(k) for k in kws)
        out[aspect] = norm.str.contains(pat, regex=True).astype(int)
    return out
